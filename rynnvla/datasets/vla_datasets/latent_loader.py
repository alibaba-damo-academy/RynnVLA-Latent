"""Multi-view latent-action slot loader for hierarchical VLA training.

Loads per-view latent actions (the 608-dim ``ktoken_zcam`` npz corpus; see the
"Latent-action protocol" section of README.md) and assembles
them into per-sample slot tensors, slot j = j-th camera in sorted name order (the same
order the processor packs images in). Path resolution is left to the dataset (which
knows its own on-disk layout); this loader only does load + normalize + slot placement,
so it is dataset-agnostic and unit-testable.

Per-view latent file (from the extractor):  <dir>/latent.npz
    latent_action  [T, latent_dim] float16
    pair_indices   [T, 2] int32
    meta           JSON string carrying the labeling protocol (pair_stride, gap, fps, ...)
Latent index ``i`` pairs source frames ``(i*P, i*P+gap)`` where ``P`` is the extractor's
``pair_stride``. The frame-aligned extractor uses ``P=1`` (``T = num_frames - gap``, one
latent per frame); RynnLAM's stride-4 labeling uses ``P=4``, giving ``T = (num_frames -
gap - 1) // P + 1``. Callers therefore need ``P`` to map a latent index back to a frame;
this loader only slices in latent-index space and leaves that mapping to the dataset.
"""

import json
import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def load_latent_stats(path: Optional[str], dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load merged latent stats (mean/std). Identity (0/1) only when no path is configured."""
    if path is None:
        return np.zeros(dim, np.float32), np.ones(dim, np.float32)
    # An explicit path that cannot be read must be loud, not identity: a typo here trains on
    # raw un-normalized latents while the dataset still logs stats='yes'.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"latent_stats_path does not exist: {path}")
    with open(path, "r") as f:
        d = json.load(f)
    mean = np.asarray(d["mean"], np.float32)
    std = np.asarray(d["std"], np.float32)
    assert mean.shape[0] == dim, f"stats dim {mean.shape[0]} != latent_action_dim {dim}"
    assert std.shape[0] == dim, f"stats dim {std.shape[0]} != latent_action_dim {dim}"
    if not bool(np.all(std > 0)):
        bad = int(np.argmin(std))
        raise ValueError(f"latent stats std must be strictly positive; std[{bad}]={std[bad]}")
    return mean, std


class LatentSlotLoader:
    """Assemble per-view latent actions into a per-sample (N, chunk, dim) slot tensor + mask.

    Args:
        latent_action_dim: latent width; 608 for the delivered `ktoken_zcam` protocol.
        stats_path: merged latent stats json (mean/std) for standardization.
        cache_size: number of per-view npz arrays to keep in an LRU cache.
    """

    def __init__(self, latent_action_dim: int,
                 stats_path: Optional[str] = None, cache_size: int = 64):
        self.dim = int(latent_action_dim)
        self.mean, self.std = load_latent_stats(stats_path, self.dim)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size

    def _load_array(self, latent_path: str) -> np.ndarray:
        """Load (and cache) the [T, dim] latent_action array from a per-view file.

        Format is dispatched by extension: ``.npz`` (the latent extractor's default) or
        ``.safetensors`` (key ``latent_action``); the RynnVLA-Base latent pack format
        is not finalized, so both are accepted.
        """
        if latent_path in self._cache:
            self._cache.move_to_end(latent_path)
            return self._cache[latent_path]
        if latent_path.endswith(".safetensors"):
            from safetensors.numpy import load_file

            d = load_file(latent_path)
            if "latent_action" not in d:
                raise KeyError(
                    f"{latent_path}: expected key 'latent_action', found {sorted(d)[:8]}"
                )
            arr = d["latent_action"]  # [T, dim], kept at the on-disk width
        else:
            # allow_pickle stays off: these files hold latent_action (float16), pair_indices
            # (int32) and a fixed-width unicode meta scalar, none of which need object
            # deserialization, so enabling it would only let a crafted file execute arbitrary
            # constructors on read.
            with np.load(latent_path, allow_pickle=False) as d:
                arr = d["latent_action"]  # [T, dim], kept at the on-disk width
        # Cached at the on-disk width, NOT widened here. Widening before the slice allocated a
        # full float32 [T, dim] copy on every miss in order to use `chunk` of its rows: at
        # BEHAVIOR-1K's median T=2512 that is 6.1 MiB allocated for 14.2 KiB used, and it
        # halved how many bytes the 64-entry LRU could hold. load_slots widens the sliced
        # chunk instead, which is bit-identical -- float16 -> float32 is exact and slicing
        # commutes with an elementwise cast.
        # The only place that sees the real array width -- build_index reads manifest metadata
        # alone. Without this, a corpus labelled with one representation (RynnLAM's ktoken_zcam
        # is 608-d) fed to a recipe configured for another reaches normalization and fails there
        # as a broadcasting error naming neither shape nor either configuration source.
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"{latent_path}: latent_action has shape {tuple(arr.shape)}, expected "
                f"[T, {self.dim}]. The corpus representation and the recipe's "
                f"latent_action_dim={self.dim} disagree; relabel with the matching "
                "representation or set latent_action_dim to the labelled width."
            )
        self._cache[latent_path] = arr
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arr

    def _chunk_from_array(self, arr: np.ndarray, start: int, chunk: int, stride: int = 1) -> np.ndarray:
        """Slice a strided latent chunk, clamping indices to repeat the last valid latent
        at episode tail. ``start`` and ``stride`` are both LATENT-index quantities: the
        chunk spans ``(chunk - 1) * stride`` latents, which is ``(chunk - 1) * stride *
        pair_stride`` source frames."""
        T = arr.shape[0]
        idx = np.clip(start + np.arange(chunk) * int(stride), 0, max(T - 1, 0))
        return arr[idx]  # [chunk, dim]

    def load_slots(
        self,
        cam_to_npz: Dict[str, str],
        start_frame: int = 0,
        chunk: int = 1,
        stride: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (latent_targets [N, chunk, dim], slot_mask [N]) for one sample.

        Slot j is the j-th camera in sorted(cam_to_npz) order — the same sorted-key
        order the processor packs images in, so latent slot j supervises the expert
        slot seeded from image j. N = len(cam_to_npz); nothing is fixed to a role
        vocabulary.

        Args:
            cam_to_npz: camera_name -> path to that camera's latent.npz (episode-specific).
            start_frame: source frame index where the chunk begins.
            chunk: latent chunk length.
            stride: source-frame stride between adjacent latent targets.
        """
        cams = sorted(cam_to_npz)
        N = len(cams)
        latent_targets = torch.zeros(N, chunk, self.dim, dtype=torch.float32)
        slot_mask = torch.ones(N, dtype=torch.bool)
        for slot, cam in enumerate(cams):
            arr = self._load_array(cam_to_npz[cam])
            chunk_arr = self._chunk_from_array(arr, start_frame, chunk, stride=stride)  # [chunk, dim]
            # Widen the sliced chunk, not the cached array -- see _load_array.
            chunk_arr = chunk_arr.astype(np.float32, copy=False)
            chunk_arr = (chunk_arr - self.mean) / self.std               # standardize
            latent_targets[slot] = torch.from_numpy(chunk_arr)
        return latent_targets, slot_mask
