import hashlib
import json
import os
from abc import ABCMeta, abstractmethod
from dataclasses import fields
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import ProcessorMixin

from ...utils.robot import (
    Arm,
    Position,
    RobotAction,
    RobotState,
    Rotation,
)
from ...constants import CACHE_DIR, RobotType, RotationRepresentation
from ...utils.logging import get_logger


logger = get_logger(__name__)


_AUG_RANGES = {
    "brightness": (0.7, 1.3),
    "contrast": (0.6, 1.4),
    "saturation": (0.5, 1.5),
}


def _sample_visual_augmentation_params() -> Dict[str, float]:
    """Sample one augmentation config shared by all camera views of a sample."""
    return {k: float(np.random.uniform(*r)) for k, r in _AUG_RANGES.items()}


def _apply_visual_augmentation(image: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    """Brightness/contrast/saturation jitter on a (H, W, C) image, lingbot-vla-v2 style."""
    orig_dtype = image.dtype
    image = image.to(torch.float32)
    if image.max() > 1.0:
        image = image / 255.0
    image = image * params["brightness"]
    mean = image.mean(dim=(0, 1, 2), keepdim=True)
    image = (image - mean) * params["contrast"] + mean
    gray = image.mean(dim=-1, keepdim=True)
    image = gray + (image - gray) * params["saturation"]
    image = image.clamp(0.0, 1.0)
    if orig_dtype == torch.uint8:
        return (image * 255.0).round().to(torch.uint8)
    return image.to(orig_dtype)


def _to_teacher_image(image, size: int = 224) -> torch.Tensor:
    """Coerce a camera frame (PIL / ndarray / tensor, CHW or HWC) to (3, size, size) in [0, 1]."""
    if isinstance(image, torch.Tensor):
        arr = image.detach()
    elif isinstance(image, np.ndarray):
        arr = torch.from_numpy(np.ascontiguousarray(image))
    else:  # PIL
        arr = torch.from_numpy(np.asarray(image.convert("RGB")).copy())
    if arr.ndim == 2:
        arr = arr.unsqueeze(-1)
    # CHW -> HWC, using the same discrimination as the processor's _to_pil.
    if arr.shape[0] in (1, 3, 4):
        arr = arr.permute(1, 2, 0)
    arr = arr.to(torch.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.shape[-1] == 1:
        arr = arr.expand(*arr.shape[:-1], 3)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"cannot interpret image with shape {tuple(arr.shape)}")
    chw = arr.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    chw = torch.nn.functional.interpolate(chw, size=(size, size), mode="bicubic", align_corners=False)
    return chw[0].clamp(0.0, 1.0)


class _EpisodeStatsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: "BaseVLADataset"):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset.episode_lengths)

    def __getitem__(self, episode_index: int):
        return self.dataset._get_episode_schemas(episode_index)


def _identity_collate(batch):
    return batch[0]


_LEAF_META_KEYS = ("type", "is_relative", "allow_relative", "representation", "dim")

# q01/q99 for action_norm_type="q01_q99". Exact quantiles are not computable from
# the single-pass sum/sum_sq/min/max accumulator above, so each leaf also keeps a
# reservoir: the _RESERVOIR_SIZE rows with the largest uniform-random keys seen so
# far. That is an exact uniform sample and, unlike binned sketches, it stays valid
# under any merge order -- merging is just concat + keep-top-K keys. Spilling at
# 2x amortizes the topk to one call per ~_RESERVOIR_SIZE new rows instead of one
# per episode. 200k rows put the q01/q99 rank error around 7e-4.
_RESERVOIR_SIZE = 200_000
_RESERVOIR_SPILL = 2 * _RESERVOIR_SIZE


def _new_accumulator(tensor: torch.Tensor) -> Dict:
    flat = tensor.reshape(-1, tensor.shape[-1]).float()
    return {
        "sum": flat.sum(0),
        "sum_sq": (flat ** 2).sum(0),
        "min": flat.amin(0),
        "max": flat.amax(0),
        "count": int(flat.size(0)),
        "res_keys": torch.rand_like(flat),
        "res_vals": flat,
    }


def _merge_reservoir(dst: Dict) -> None:
    """Trim ``dst``'s reservoir back to _RESERVOIR_SIZE rows once it passes the spill cap."""
    keys, vals = dst["res_keys"], dst["res_vals"]
    if keys.size(0) <= _RESERVOIR_SPILL:
        return
    top = torch.topk(keys, _RESERVOIR_SIZE, dim=0)
    dst["res_keys"] = top.values
    dst["res_vals"] = vals.gather(0, top.indices)


def _consolidate_schema(schema: Dict) -> None:
    """Trim every leaf reservoir in place before the cross-rank gather.

    Without this the all_gather_object payload carries up to _RESERVOIR_SPILL rows
    per leaf per rank; the reservoir is discarded by _finalize_leaf anyway, so it
    must never reach the schema cache or processor_config.json.
    """
    for k, v in schema.items():
        if k == "type":
            continue
        if _is_leaf(v):
            keys = v["res_keys"]
            if keys.size(0) > _RESERVOIR_SIZE:
                top = torch.topk(keys, _RESERVOIR_SIZE, dim=0)
                v["res_keys"] = top.values
                v["res_vals"] = v["res_vals"].gather(0, top.indices)
        else:
            _consolidate_schema(v)


def _merge_accumulator(dst: Dict, src: Dict) -> None:
    dst["sum"] = dst["sum"] + src["sum"]
    dst["sum_sq"] = dst["sum_sq"] + src["sum_sq"]
    dst["min"] = torch.minimum(dst["min"], src["min"])
    dst["max"] = torch.maximum(dst["max"], src["max"])
    dst["count"] = dst["count"] + src["count"]
    dst["res_keys"] = torch.cat([dst["res_keys"], src["res_keys"]], dim=0)
    dst["res_vals"] = torch.cat([dst["res_vals"], src["res_vals"]], dim=0)
    _merge_reservoir(dst)


def _finalize_leaf(leaf: Dict) -> Dict:
    count = leaf["count"]
    mean = leaf["sum"] / count
    if count > 1:
        var = (leaf["sum_sq"] - leaf["sum"] ** 2 / count) / (count - 1)
    else:
        var = torch.zeros_like(mean)
    std = torch.sqrt(var.clamp(min=0))
    keys, vals = leaf["res_keys"], leaf["res_vals"]
    if keys.size(0) > _RESERVOIR_SIZE:
        top = torch.topk(keys, _RESERVOIR_SIZE, dim=0)
        vals = vals.gather(0, top.indices)
    q = torch.quantile(vals, torch.tensor([0.01, 0.99], dtype=vals.dtype), dim=0)
    out = {mk: leaf[mk] for mk in _LEAF_META_KEYS}
    out["mean"] = mean.tolist()
    out["std"] = std.tolist()
    out["min"] = leaf["min"].tolist()
    out["max"] = leaf["max"].tolist()
    out["q01"] = q[0].tolist()
    out["q99"] = q[1].tolist()
    out["count"] = count
    return out


def _leaf_to_cpu(leaf: Dict) -> Dict:
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in leaf.items()}


def _is_leaf(value) -> bool:
    return isinstance(value, dict) and "dim" in value


def _atomic_leaf(value) -> Dict:
    """Build a single leaf dict with structure metadata and accumulator stats
    inlined at the same level."""
    if isinstance(value, Position):
        meta = {
            "type": type(value).__name__,
            "is_relative": bool(value.is_relative),
            "allow_relative": bool(value.allow_relative),
            "representation": None,
            "dim": int(value.data.size(-1)),
        }
    elif isinstance(value, Rotation):
        meta = {
            "type": type(value).__name__,
            "is_relative": bool(value.is_relative),
            "allow_relative": bool(value.allow_relative),
            "representation": value.representation.value,
            "dim": int(value.data.size(-1)),
        }
    else:
        raise TypeError(f"Expected Position or Rotation, got {type(value).__name__}")
    return {**meta, **_new_accumulator(value.data)}


def _field_schema(value) -> Dict:
    """Build schema for one top-level RobotAction field."""
    if isinstance(value, (Position, Rotation)):
        return _atomic_leaf(value)
    if isinstance(value, Arm):
        out: Dict = {"type": type(value).__name__}
        for f in fields(value):
            v = getattr(value, f.name)
            if v is None:
                continue
            out[f.name] = _atomic_leaf(v)
        return out
    raise TypeError(f"Unsupported field type: {type(value).__name__}")


def _robot_schema(robot_action: RobotAction) -> Dict:
    """Build a full schema tree for a RobotAction/RobotState."""
    out: Dict = {"type": type(robot_action).__name__}
    for f in fields(robot_action):
        v = getattr(robot_action, f.name)
        if v is None:
            continue
        out[f.name] = _field_schema(v)
    return out


def _merge_schema(dst: Dict, src: Dict, _path: str = "") -> None:
    """Recursively merge ``src`` schema into ``dst`` in place.

    Enforces structural consistency for the same robot type: field sets and leaf
    metadata must match across episodes; mismatches are raised with the offending
    path so divergent dataset schemas surface early.
    """
    if set(dst.keys()) != set(src.keys()):
        only_dst = sorted(set(dst.keys()) - set(src.keys()))
        only_src = sorted(set(src.keys()) - set(dst.keys()))
        raise ValueError(
            f"Schema field set mismatch at '{_path or '<root>'}': "
            f"only_in_existing={only_dst}, only_in_new={only_src}"
        )
    for k in dst:
        d, s = dst[k], src[k]
        path = f"{_path}.{k}" if _path else k
        if k == "type":
            if d != s:
                raise ValueError(
                    f"Schema type mismatch at '{path}': {d!r} != {s!r}"
                )
            continue
        if _is_leaf(d):
            if not _is_leaf(s):
                raise ValueError(
                    f"Schema mismatch at '{path}': existing is a leaf, new is a sub-dict"
                )
            for mk in _LEAF_META_KEYS:
                if d[mk] != s[mk]:
                    raise ValueError(
                        f"Schema metadata mismatch at '{path}.{mk}': "
                        f"{d[mk]!r} != {s[mk]!r}"
                    )
            _merge_accumulator(d, s)
        else:
            if _is_leaf(s):
                raise ValueError(
                    f"Schema mismatch at '{path}': existing is a sub-dict, new is a leaf"
                )
            _merge_schema(d, s, _path=path)


def _finalize_schema(schema: Dict) -> Dict:
    out: Dict = {}
    for k, v in schema.items():
        if k == "type":
            out[k] = v
            continue
        if _is_leaf(v):
            out[k] = _finalize_leaf(v)
        else:
            out[k] = _finalize_schema(v)
    return out


def _schema_to_cpu(schema: Dict) -> Dict:
    out: Dict = {}
    for k, v in schema.items():
        if k == "type":
            out[k] = v
            continue
        if _is_leaf(v):
            out[k] = _leaf_to_cpu(v)
        else:
            out[k] = _schema_to_cpu(v)
    return out


class BaseVLADataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        data_path: str,
        action_chunk_size: int,
        use_delta_action: bool,
        eef_rotation_repr: Optional[RotationRepresentation] = None,
        target_fps: Optional[float] = None,
        processor: Optional[ProcessorMixin] = None,
        use_visual_augmentation: bool = False,
        chunk_overlap_ratio: float = 0.0,
        **kwargs,
    ):
        assert action_chunk_size > 0
        assert eef_rotation_repr is None or isinstance(eef_rotation_repr, RotationRepresentation)
        assert target_fps is None or target_fps > 0
        assert 0.0 <= chunk_overlap_ratio < 1.0, "chunk_overlap_ratio must be in [0, 1)"

        self.data_path = data_path
        self.action_chunk_size = action_chunk_size
        self.use_delta_action = use_delta_action
        self.eef_rotation_repr = eef_rotation_repr
        self.target_fps = target_fps
        self.processor = processor
        self.num_view_slots = int(kwargs.get("num_view_slots", 1))
        # SF alignment: attach un-augmented 224x224 copies of every camera frame as
        # teacher input (batch-ordered like image_grid_thw; sorted-key camera order,
        # matching the processor's image layout).
        self.emit_teacher_images = bool(kwargs.get("emit_teacher_images", False))

        # Chunk overlap: stride = chunk_size * (1 - overlap_ratio)
        # e.g., chunk_size=50, overlap_ratio=0.5 -> stride=25 (50% overlap)
        self.chunk_stride = max(1, int(action_chunk_size * (1.0 - chunk_overlap_ratio)))

        # Visual augmentation (brightness/contrast/saturation, lingbot-vla-v2 style)
        self.visual_augmentation = use_visual_augmentation

        src_lengths = np.asarray(self.episode_lengths, dtype=np.int64)
        if target_fps is None:
            self._target_lengths = src_lengths
        else:
            target_lengths = np.empty_like(src_lengths)
            for i, n in enumerate(src_lengths.tolist()):
                src_fps = float(self.get_fps(i))
                if abs(src_fps - target_fps) < 1e-6:
                    target_lengths[i] = int(n)
                else:
                    target_lengths[i] = max(1, int(round(n * target_fps / src_fps)))
            self._target_lengths = target_lengths

        self._cum_lengths = np.cumsum(self._index_lengths())

    def _index_lengths(self) -> np.ndarray:
        """Per-episode size of the index space that ``__len__`` / ``_resolve_index`` enumerate.

        Defaults to one index per TARGET FRAME. A subclass whose ``__getitem__`` snaps an index
        onto a coarser grid must override this, otherwise ``len()`` overcounts: the same distinct
        sample is handed back several times per epoch, ``num_update_steps_per_epoch`` is inflated
        by the same factor, and the resume batch-skip math divides by the wrong epoch length.

        ``LatentPretrainDataset`` needs no override: it already returns chunk-start counts from
        ``episode_lengths``, so its ``_target_lengths`` is in start space to begin with.
        """
        return self._target_lengths

    def _chunk_start_counts(self) -> np.ndarray:
        """Distinct chunk starts per episode, matching what ``__getitem__`` actually produces.

        For a target length ``L``, chunk ``K`` and stride ``S`` the clamp ceiling is
        ``m = max(0, L - K)``. The reachable starts are the multiples of ``S`` up to ``m``, plus
        ``m`` itself when it is not already one of them (indices past the last multiple all clamp
        onto it). An episode shorter than one chunk yields exactly one padded sample.
        """
        lengths = self._target_lengths.astype(np.int64)
        stride = max(1, int(self.chunk_stride))
        ceiling = np.maximum(0, lengths - int(self.action_chunk_size))
        counts = ceiling // stride + 1 + (ceiling % stride != 0).astype(np.int64)
        return np.where(lengths <= int(self.action_chunk_size), 1, counts)

    @property
    @abstractmethod
    def episode_lengths(self) -> List[int]:
        ...

    def get_fps(self, episode_index: int) -> float:
        """Return the source fps of `episode_index`.

        Subclasses must override this when `target_fps` is requested. The default
        implementation raises so backward-compatible (``target_fps=None``) usage
        keeps working without subclass changes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_fps to support target_fps"
        )

    @abstractmethod
    def get_robot_type(self, episode_index: int) -> RobotType:
        ...

    @abstractmethod
    def load_action(
        self,
        episode_index: int,
        frame_index: Union[int, List[int], slice],
    ) -> RobotAction:
        ...

    @abstractmethod
    def load_state(
        self,
        episode_index: int,
        frame_index: Union[int, List[int], slice],
    ) -> RobotState:
        ...

    @abstractmethod
    def load_images(
        self,
        episode_index: int,
        frame_index: Union[int, List[int], slice],
    ) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def load_instruction(
        self,
        episode_index: int,
        frame_index: Union[int, List[int], slice],
    ) -> List[str]:
        ...

    @abstractmethod
    def load_episode(self, episode_index: int, action_only: bool) -> Dict:
        ...

    def __len__(self) -> int:
        return int(self._cum_lengths[-1]) if len(self._cum_lengths) > 0 else 0

    def _resolve_index(self, index: int) -> Tuple[int, int, int]:
        """Map a global index → ``(episode_index, ordinal, target_episode_length)``.

        ``ordinal`` is a position in the episode's slice of the index space defined by
        ``_index_lengths`` -- a chunk start ordinal for a dataset that overrides it, a frame
        index for one that does not. The third element is always the TARGET FRAME length, not the
        index-space slice size: ``__getitem__`` needs it to clamp the chunk and bound
        ``target_end``.
        """
        cum = self._cum_lengths
        episode_index = int(np.searchsorted(cum, index, side="right"))
        episode_start = int(cum[episode_index - 1]) if episode_index > 0 else 0
        return episode_index, index - episode_start, int(self._target_lengths[episode_index])

    def _target_to_source_index(self, episode_index: int, target_frame: int) -> int:
        """Nearest source-frame index for a single target frame."""
        if self.target_fps is None:
            return target_frame
        src_fps = float(self.get_fps(episode_index))
        if abs(src_fps - self.target_fps) < 1e-6:
            return target_frame
        src_n = int(self.episode_lengths[episode_index])
        ratio = src_fps / self.target_fps
        return min(max(0, int(round(target_frame * ratio))), src_n - 1)

    def _target_chunk_to_source(
        self, episode_index: int, target_start: int, target_end: int,
    ) -> Tuple[slice, Optional[Dict]]:
        """Map a target-fps chunk [target_start, target_end) to a covering source slice
        and the kwargs needed for ``Action.resample`` to land on the requested target chunk.

        Returns ``(src_slice, None)`` when no resampling is required.
        """
        if self.target_fps is None:
            return slice(target_start, target_end), None
        src_fps = float(self.get_fps(episode_index))
        if abs(src_fps - self.target_fps) < 1e-6:
            return slice(target_start, target_end), None

        src_n = int(self.episode_lengths[episode_index])
        ratio = src_fps / self.target_fps
        s0 = max(0, int(np.floor(target_start * ratio)))
        s1 = int(np.ceil((target_end - 1) * ratio)) + 2  # +2: ceil-exclusive end + linear tail
        s1 = min(src_n, max(s1, s0 + 1))
        return slice(s0, s1), {
            "src_fps": src_fps,
            "tgt_fps": float(self.target_fps),
            "n_target": target_end - target_start,
            "src_offset": target_start * ratio - s0,
        }

    def __getitem__(self, index: int):
        episode_index, ordinal, target_n = self._resolve_index(index)

        # `ordinal` is already a chunk-start ordinal (see _index_lengths / _chunk_start_counts),
        # so the start is ordinal * chunk_stride clamped to the last full chunk. This yields the
        # same set of starts the old frame-granular index space produced, but each exactly once
        # instead of chunk_stride times -- the snap-and-clamp it replaces is what made len()
        # overcount and every distinct chunk come back several times per epoch.
        target_frame = min(ordinal * self.chunk_stride, max(0, target_n - self.action_chunk_size))

        target_end = min(target_frame + self.action_chunk_size, target_n)
        src_slice, resample_kwargs = self._target_chunk_to_source(
            episode_index, target_frame, target_end,
        )
        state_src_idx = self._target_to_source_index(episode_index, target_frame)

        action = self.load_action(episode_index, src_slice)
        state = self.load_state(episode_index, state_src_idx)

        if resample_kwargs is not None:
            action = action.resample(**resample_kwargs)

        if len(action) < self.action_chunk_size:
            action = action.pad_to(self.action_chunk_size)

        if self.eef_rotation_repr is not None:
            action = action.convert_rotation(self.eef_rotation_repr)
            state = state.convert_rotation(self.eef_rotation_repr)

        if self.use_delta_action:
            action = action - state

        outputs = {
            "robot_type": self.get_robot_type(episode_index),
            "action": action,
            "state": state,
            "text": self.load_instruction(episode_index, state_src_idx)[0],
            "images": {k: v[0] for k, v in self.load_images(episode_index, state_src_idx).items()},
        }

        camera_slot_map = getattr(self, "camera_slot_map", None)
        if camera_slot_map is not None:
            # slot_mask marks which of this sample's cameras are real; every camera the
            # dataset yielded is, so it is all-ones of length N, padded to the batch max
            # N by the collator. NOTE this is still the sample-order (dense) convention,
            # used only by the latent multi-view path. camera_slot_ids, which the
            # processor derives from camera_slot_map below, is NOT sample-order any more:
            # it carries constants.VIEW_ROLES ids. Reconciling slot_mask onto the fixed
            # role axis belongs to the latent-pretraining change, not here.
            outputs["slot_mask"] = torch.ones(len(outputs["images"]), dtype=torch.bool)
            outputs["camera_slot_map"] = camera_slot_map

        teacher_images = None
        primary_teacher_index = None
        if self.emit_teacher_images:
            camera_keys = sorted(outputs["images"])
            if not camera_keys:
                raise ValueError("emit_teacher_images=True requires at least one camera image")
            teacher_images = torch.stack([
                _to_teacher_image(outputs["images"][key]) for key in camera_keys
            ])
            primary_camera_key = getattr(self, "primary_camera_key", camera_keys[0])
            if primary_camera_key not in camera_keys:
                raise ValueError(
                    f"primary_camera_key={primary_camera_key!r} is absent; cameras={camera_keys}"
                )
            primary_teacher_index = camera_keys.index(primary_camera_key)

        # Visual augmentation (brightness/contrast/saturation)
        if self.visual_augmentation:
            aug_params = _sample_visual_augmentation_params()
            outputs["images"] = {
                k: _apply_visual_augmentation(v, aug_params) for k, v in outputs["images"].items()
            }

        if self.processor is None:
            if teacher_images is not None:
                outputs["teacher_images"] = teacher_images
                outputs["primary_teacher_index"] = torch.tensor(primary_teacher_index)
            return outputs

        features = self.processor(**outputs, return_tensors="pt")
        if teacher_images is not None:
            image_grid_thw = features.get("image_grid_thw")
            if image_grid_thw is None or image_grid_thw.size(0) < teacher_images.size(0):
                shape = None if image_grid_thw is None else tuple(image_grid_thw.shape)
                raise ValueError(
                    f"teacher images require matching image grids; teacher={teacher_images.size(0)}, "
                    f"image_grid_thw={shape}"
                )
            grid_offset = image_grid_thw.size(0) - teacher_images.size(0)
            features["teacher_images"] = teacher_images
            features["teacher_image_grid_indices"] = torch.arange(
                grid_offset, image_grid_thw.size(0), dtype=torch.long
            )
            features["primary_teacher_index"] = torch.tensor(primary_teacher_index, dtype=torch.long)
        return features

    def _get_episode_schemas(
        self, episode_index: int
    ) -> Tuple[RobotType, Dict, Dict]:
        episode = self.load_episode(episode_index, action_only=True)
        action: RobotAction = episode["action"]
        state: RobotState = episode["state"]
        robot_type: RobotType = self.get_robot_type(episode_index)

        target_n = int(self._target_lengths[episode_index])
        if self.target_fps is not None and len(action) != target_n:
            src_fps = float(self.get_fps(episode_index))
            action = action.resample(src_fps, float(self.target_fps), n_target=target_n)
            state = state.resample(src_fps, float(self.target_fps), n_target=target_n)

        N = len(action)
        K = self.action_chunk_size

        if self.eef_rotation_repr is not None:
            action = action.convert_rotation(self.eef_rotation_repr)
            state = state.convert_rotation(self.eef_rotation_repr)

        # State: one accumulator pass over the whole episode.
        state_schema = _robot_schema(state)

        # Action: aggregate over chunks. After per-chunk subtraction the
        # is_relative metadata may switch; capture structure from the first
        # chunk and merge subsequent chunks' accumulators into it.
        chunk0 = action[0:min(K, N)]
        if len(chunk0) < K:
            chunk0 = chunk0.pad_to(K)
        if self.use_delta_action:
            chunk0 = chunk0 - state[0:1]
        action_schema = _robot_schema(chunk0)

        for i in range(1, N):
            end = min(i + K, N)
            chunk = action[i:end]
            if len(chunk) < K:
                chunk = chunk.pad_to(K)
            if self.use_delta_action:
                chunk = chunk - state[i:i + 1]
            _merge_schema(action_schema, _robot_schema(chunk))

        return robot_type, action_schema, state_schema

    def _schema_cache_key(self) -> Dict:
        return {
            "data_path": os.path.abspath(self.data_path),
            "action_chunk_size": self.action_chunk_size,
            "use_delta_action": self.use_delta_action,
            "eef_rotation_repr": (
                self.eef_rotation_repr.value
                if self.eef_rotation_repr is not None
                else None
            ),
            "target_fps": self.target_fps,
            "num_episodes": len(self.episode_lengths),
            "num_frames": int(self._target_lengths.sum()),
            "quantiles": [0.01, 0.99],
        }

    def _schema_cache_path(self, key: Dict) -> str:
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True).encode("utf-8")
        ).hexdigest()
        # Separate namespace from the pre-quantile "schemas" cache: a cached leaf
        # written before q01/q99 existed has no such keys, and reusing it would make
        # _norm_coeffs raise KeyError("q01") on the first normalized sample.
        return os.path.join(CACHE_DIR, "schemas_q01q99", f"{self.__class__.__name__}_{digest}.json")

    def _balanced_episode_assignment(self, world_size: int) -> List[List[int]]:
        episode_lengths = list(self.episode_lengths)
        sorted_eps = sorted(
            range(len(episode_lengths)),
            key=lambda i: episode_lengths[i],
            reverse=True,
        )
        loads = [0] * world_size
        assignments: List[List[int]] = [[] for _ in range(world_size)]
        for ep_idx in sorted_eps:
            min_w = min(range(world_size), key=lambda w: loads[w])
            assignments[min_w].append(ep_idx)
            loads[min_w] += episode_lengths[ep_idx]
        return assignments

    def get_schema(
        self,
        num_workers: int = 8,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Dict[str, Dict]:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank(group=process_group)
            world_size = torch.distributed.get_world_size(group=process_group)
        else:
            rank = 0
            world_size = 1

        cache_key = self._schema_cache_key()
        cache_path = self._schema_cache_path(cache_key)

        cache_hit = False
        cached_result: Optional[Dict[str, Dict]] = None
        if rank == 0 and os.path.isfile(cache_path):
            logger.info(f"Loading cached schema from {cache_path}")
            with open(cache_path, "r") as f:
                cached = json.load(f)
            cached_result = {"action": cached["action"], "state": cached["state"]}
            cache_hit = True

        if world_size > 1:
            obj_list = [cache_hit]
            torch.distributed.broadcast_object_list(
                obj_list, group=process_group, group_src=0
            )
            cache_hit = obj_list[0]
            if cache_hit:
                obj_list = [cached_result]
                torch.distributed.broadcast_object_list(
                    obj_list, group=process_group, group_src=0
                )
                cached_result = obj_list[0]

        if cache_hit:
            return cached_result

        if world_size > 1:
            assignments = self._balanced_episode_assignment(world_size)
            local_episodes = sorted(assignments[rank])
        else:
            local_episodes = list(range(len(self.episode_lengths)))

        subset = torch.utils.data.Subset(_EpisodeStatsDataset(self), local_episodes)
        dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=_identity_collate,
            shuffle=False,
        )

        action_local: Dict[str, Dict] = {}
        state_local: Dict[str, Dict] = {}
        for robot_type, ep_action, ep_state in tqdm(
            dataloader, desc="Computing schemas", disable=rank > 0
        ):
            rt = robot_type.value
            if rt not in action_local:
                action_local[rt] = ep_action
            else:
                _merge_schema(action_local[rt], ep_action)
            if rt not in state_local:
                state_local[rt] = ep_state
            else:
                _merge_schema(state_local[rt], ep_state)

        if world_size > 1:
            for s in list(action_local.values()) + list(state_local.values()):
                _consolidate_schema(s)
            payload = {
                "action": {rt: _schema_to_cpu(s) for rt, s in action_local.items()},
                "state": {rt: _schema_to_cpu(s) for rt, s in state_local.items()},
            }
            gathered: List[Optional[Dict]] = [None] * world_size
            torch.distributed.all_gather_object(
                gathered, payload, group=process_group
            )
            action_merged: Dict[str, Dict] = {}
            state_merged: Dict[str, Dict] = {}
            for partial in gathered:
                if not partial:
                    continue
                for rt, s in partial["action"].items():
                    if rt not in action_merged:
                        action_merged[rt] = s
                    else:
                        _merge_schema(action_merged[rt], s)
                for rt, s in partial["state"].items():
                    if rt not in state_merged:
                        state_merged[rt] = s
                    else:
                        _merge_schema(state_merged[rt], s)
            action_local, state_local = action_merged, state_merged

        result: Dict[str, Dict] = {
            "action": {rt: _finalize_schema(s) for rt, s in action_local.items()},
            "state": {rt: _finalize_schema(s) for rt, s in state_local.items()},
        }

        if rank == 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"metadata": cache_key, **result}, f, indent=4)

        if world_size > 1:
            torch.distributed.barrier(group=process_group)

        return result
