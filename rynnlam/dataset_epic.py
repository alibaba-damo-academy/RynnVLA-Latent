"""
LAM Dataset that reads directly from the original JSON + processed data directories.

No intermediate organization step needed. Just point to:
  - meta JSON: short_cap_v1.json (list of video paths)
  - depth_root: where Depth-Anything-3 saved results.npz
  - flow_root: where ptlflow saved flow.mp4 and/or flows.npz

Data structure (as processed by your scripts):
    depth_root/<sub_dir>/exports/mini_npz/results.npz  -> depth, intrinsics, extrinsics
    flow_root/<sub_dir>/flows.npz                      -> raw float32 optical flow (preferred)
    flow_root/<sub_dir>/flow.mp4                       -> optical flow visualization (fallback)
    <video_path from JSON>                             -> original RGB video
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import hashlib
import json
import os
import random
import struct
import threading
import zipfile

import cv2
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import load_file as safetensors_load_file
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from .logger import logger


# =============================================================================
# Online flow decoder: decode flow.mp4 (color-coded visualization) to (u,v) flow
# =============================================================================

_DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def _make_colorwheel():
    """Recreate the colorwheel used in ptlflow's flow_to_rgb encoding."""
    colorwheel_length = sum(_DEFAULT_TRANSITIONS)
    base_hues = [
        [255, 0, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
        [255, 0, 255],
        [255, 0, 0],
    ]
    colorwheel = np.zeros((colorwheel_length, 3), dtype="float32")
    hue_from = base_hues[0]
    start_index = 0
    for i in range(len(_DEFAULT_TRANSITIONS)):
        end_index = start_index + _DEFAULT_TRANSITIONS[i]
        hue_to = base_hues[i + 1]
        for c in range(3):
            colorwheel[start_index:end_index, c] = np.linspace(
                hue_from[c], hue_to[c], _DEFAULT_TRANSITIONS[i], endpoint=False
            )
        hue_from = hue_to
        start_index = end_index
    return colorwheel


_COLORWHEEL = None
_NCOLS = None


def _decode_flow_frame(
    rgb_frame: np.ndarray, flow_max_radius: float = 20.0
) -> np.ndarray:
    """
    Decode a single RGB flow frame (from flow.mp4) back to approximate (u,v) flow.
    """
    global _COLORWHEEL, _NCOLS
    if _COLORWHEEL is None:
        _COLORWHEEL = _make_colorwheel()
        _NCOLS = len(_COLORWHEEL)

    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[:, :, 0]
    # flow.mp4 is encoded with flowpy's 'bright' mode: RGB = 255 - radius*(255-hue),
    # equivalent to HSV with S=radius and V fixed at 255. Displacement magnitude lives
    # in the [saturation S] channel, not V.
    s = hsv[:, :, 1]

    angle = (h / 179.0) * (_NCOLS - 1)
    angle_rad = (angle / (_NCOLS - 1)) * 2 * np.pi
    angle_rad = np.where(angle_rad > np.pi, angle_rad - 2 * np.pi, angle_rad)

    radius_norm = np.clip(s / 255.0, 0, 1)
    radius = radius_norm * flow_max_radius

    u = radius * np.cos(angle_rad)
    v_flow = radius * np.sin(angle_rad)
    return np.stack([u, v_flow], axis=-1).astype(np.float32)


def _decode_flow_video(flow_mp4_path: str, flow_max_radius: float = 20.0) -> np.ndarray:
    """Decode an entire flow.mp4 video into a flow array [N, H, W, 2]."""
    cap = cv2.VideoCapture(flow_mp4_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {flow_mp4_path}")

    all_flows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flow = _decode_flow_frame(frame_rgb, flow_max_radius=flow_max_radius)
        all_flows.append(flow)
    cap.release()

    if len(all_flows) == 0:
        raise ValueError(f"No frames decoded from {flow_mp4_path}")

    return np.stack(all_flows, axis=0)


def _chain_flow_sequence(flows: np.ndarray) -> np.ndarray:
    """Chain a sequence of consecutive 2D flows into one accumulated flow.

    Args:
        flows: [K, H, W, 2] array of consecutive flows (flow[i] = flow from frame i to i+1)

    Returns:
        [H, W, 2] accumulated flow from first to last frame
    """
    if len(flows) == 1:
        return flows[0]

    fH, fW = flows[0].shape[:2]
    ys, xs = np.mgrid[:fH, :fW].astype(np.float32)
    cur_x = xs.copy()
    cur_y = ys.copy()

    for flow_i in flows:
        sx = np.clip(cur_x, 0, fW - 1)
        sy = np.clip(cur_y, 0, fH - 1)

        x0 = np.floor(sx).astype(np.int32)
        y0 = np.floor(sy).astype(np.int32)
        x1 = np.minimum(x0 + 1, fW - 1)
        y1 = np.minimum(y0 + 1, fH - 1)
        wx = sx - x0
        wy = sy - y0

        # Vectorized bilinear interpolation for both channels at once
        w00 = ((1 - wx) * (1 - wy))[..., None]
        w01 = (wx * (1 - wy))[..., None]
        w10 = ((1 - wx) * wy)[..., None]
        w11 = (wx * wy)[..., None]

        interp = (
            flow_i[y0, x0] * w00
            + flow_i[y0, x1] * w01
            + flow_i[y1, x0] * w10
            + flow_i[y1, x1] * w11
        )

        cur_x += interp[..., 0]
        cur_y += interp[..., 1]

    return np.stack([cur_x - xs, cur_y - ys], axis=-1).astype(np.float32)


def _get_sub_dir(video_path: str) -> str:
    """Extract scene directory path from video path.

    EPIC:   /.../split_videos/P01_01/P01_01_102_put cup into cupboard.mp4
            -> P01_01/P01_01_102_put cup into cupboard
    EgoDex: /.../egodex_unzipped/extra/extra/assemble_disassemble_jigsaw_puzzle/105.mp4
            -> extra/extra/assemble_disassemble_jigsaw_puzzle/105
    """
    for keyword in (
        "split_videos/",
        "egodex_unzipped/",
        "video/RDT-1B/",
        "RoboCOIN/",
        "RoboMIND/",
    ):
        if keyword in video_path:
            rel = video_path.split(keyword)[-1]
            p = Path(rel)
            return str(p.parent / p.stem)

    # Fallback: strip extension from full path
    p = Path(video_path)
    return str(p.parent / p.stem)


class _WorkerLRUCache:
    """Per-worker LRU cache for npz data arrays.

    Each DataLoader worker gets its own cache (via threading.local) to avoid
    redundant decompression when SceneGroupedBatchSampler puts multiple samples
    from the same scene into one batch.
    """

    def __init__(self, max_items: int = 8):
        self._local = threading.local()
        self._max_items = max_items

    def _get_store(self) -> OrderedDict:
        if not hasattr(self._local, "store"):
            self._local.store = OrderedDict()
        return self._local.store

    def get(self, key: str):
        store = self._get_store()
        if key in store:
            store.move_to_end(key)
            return store[key]
        return None

    def put(self, key: str, value):
        store = self._get_store()
        if key in store:
            store.move_to_end(key)
        else:
            if len(store) >= self._max_items:
                store.popitem(last=False)
            store[key] = value


def _bounded_scan_futures(pool, scan, scene_ids, limit):
    """Submit bounded chunks rather than retaining one Future per scene."""
    for start in range(0, len(scene_ids), limit):
        futures = [pool.submit(scan, sid) for sid in scene_ids[start : start + limit]]
        yield from as_completed(futures)


class LAMEpicDataset(Dataset):
    """
    Dataset that reads directly from JSON meta + processed data directories.

    Args:
        meta_json: Path to short_cap_v1.json
        depth_root: Path to Depth-Anything-3 output (e.g., Embodied-Data/Depth/Epic)
        flow_root: Path to ptlflow output (e.g., Embodied-Data/depth/EpicKitchens)
        safetensors_root: Path to the preprocessed scene-safetensors directory. When a scene's
            .safetensors file exists there, raw video/npz IO is skipped entirely.
        flow_max_radius: Radius assumption for flow.mp4 decoding
        depth_grad_threshold: Threshold for depth gradient masking
    """

    def __init__(
        self,
        meta_json: Optional[str] = None,
        depth_root: Optional[str] = None,
        flow_root: Optional[str] = None,
        safetensors_root: Optional[str] = None,
        flow_max_radius: float = 20.0,
        depth_grad_threshold: float = 0.02,
        scene_filter: Optional[List[str]] = None,
        max_scenes: int = 0,
        max_frame_stride: int = 5,
        min_frame_stride: int = 1,
        max_sample_stride: Optional[int] = None,
        target_hw: Optional[Tuple[int, int]] = None,
        use_additivity: bool = False,
        skip_frame_scan: bool = False,
        cache_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ):
        self.depth_root = Path(depth_root) if depth_root else None
        self.flow_root = Path(flow_root) if flow_root else None
        self.safetensors_root = Path(safetensors_root) if safetensors_root else None
        self.flow_max_radius = flow_max_radius
        self.depth_grad_threshold = depth_grad_threshold
        self.skip_frame_scan = skip_frame_scan
        self.max_scenes = max_scenes
        self.max_frame_stride = max_frame_stride
        self.min_frame_stride = min_frame_stride
        self.max_sample_stride = (
            max_sample_stride if max_sample_stride is not None else max_frame_stride
        )
        self.target_hw = target_hw
        self.use_additivity = use_additivity

        self._st_meta: Dict[str, dict] = {}
        self._is_sharded = False
        manifest_root = Path(manifest_path) if manifest_path else self.safetensors_root
        manifests = []
        if manifest_root:
            if manifest_root.is_file():
                manifests = [manifest_root]
            elif manifest_root.is_dir():
                manifests = sorted(
                    set(manifest_root.glob("manifest_*.json"))
                    | set(manifest_root.glob("*/manifest_*.json"))
                    | set(manifest_root.glob("manifest.json"))
                )
            if manifest_path and not manifests:
                raise FileNotFoundError(f"No manifests at {manifest_root}")
        for mf in manifests:
            with mf.open() as handle:
                entries = json.load(handle)
            if not isinstance(entries, list):
                raise ValueError(f"Manifest must contain a list: {mf}")
            for entry in entries:
                entry = dict(entry)
                reference = entry.get("shard") or entry["file"]
                reference = str((mf.parent / reference).resolve())
                entry["file"] = reference
                if "shard" in entry:
                    entry["shard"] = reference
                    self._is_sharded = True
                previous = self._st_meta.get(entry["scene_id"])
                if previous is not None and previous != entry:
                    raise ValueError(
                        f"Conflicting entries for {entry['scene_id']}; rebuild the combined manifest"
                    )
                self._st_meta[entry["scene_id"]] = entry
        if manifests:
            logger.info(
                f"Safetensors index: {len(self._st_meta)} scenes from "
                f"{len(manifests)} manifests in {manifest_root}"
                + (" (sharded)" if self._is_sharded else "")
            )

        # Build scene list. Two JSON formats supported:
        #   (A) Legacy EPIC: item = {"path": "<rgb>"} + depth_root/flow_root provided
        #       → paths constructed via _get_sub_dir()
        #   (B) Explicit: item = {"rgb": "<abs>", "depth_npz": "<abs>", "flow_npz": "<abs>",
        #                          "scene_id"?: str}  ← no roots needed
        #   (C) Safetensors-only: meta_json is None and manifests are present
        #       → scenes loaded directly from manifest_*.json files
        self.scene_list = []
        self.scene_meta = {}

        if not meta_json:
            if not self._st_meta:
                raise ValueError(
                    "meta_json is required when safetensors_root is not provided or contains no manifests."
                )
            logger.info("Building scene list from safetensors manifests...")
            for scene_id_safe, e in self._st_meta.items():
                if scene_filter and scene_id_safe not in scene_filter:
                    continue
                self.scene_list.append(scene_id_safe)
                self.scene_meta[scene_id_safe] = {
                    "video_path": "",
                    "depth_npz": "",
                    "flow_npz": "",
                    "flow_mp4": "",
                    "original_rel": scene_id_safe,
                }
            # Sort by shard file for locality (consecutive indices → same shard → page cache hits)
            if self._is_sharded:
                self.scene_list.sort(
                    key=lambda sid: self._st_meta[sid].get("shard", "")
                )
                logger.info("Scene list sorted by shard_id for locality")
        else:
            with open(meta_json, "r", encoding="utf-8") as f:
                all_items = json.load(f)

            logger.info("Building scene list from JSON...")

            for item in tqdm(all_items, desc="Parsing JSON"):
                # Mode B: explicit absolute paths in JSON
                if "rgb" in item and "depth_npz" in item:
                    video_path = item["rgb"]
                    depth_npz = item["depth_npz"]
                    flow_npz = item.get("flow_npz", "")
                    flow_mp4 = item.get("flow_mp4", "")
                    scene_id_safe = item.get("scene_id") or Path(video_path).stem
                    scene_id_safe = scene_id_safe.replace(" ", "_")
                    scene_rel = scene_id_safe  # used as cache identifier only

                    if scene_filter and scene_id_safe not in scene_filter:
                        continue
                    if scene_id_safe in self.scene_meta:
                        continue  # skip duplicates

                    self.scene_list.append(scene_id_safe)
                    self.scene_meta[scene_id_safe] = {
                        "video_path": video_path,
                        "depth_npz": str(depth_npz),
                        "flow_npz": str(flow_npz),
                        "flow_mp4": str(flow_mp4),
                        "original_rel": scene_rel,
                        # Written into the manifest by preprocess; the training-side
                        # WeightedMultiDatasetBatchSampler uses it for per-dataset sampling
                        # weights. Falls back to "unknown" when an old meta lacks the field.
                        "dataset": item.get("dataset", "unknown"),
                        "role": item.get("role", scene_id_safe.rsplit("/", 1)[-1]),
                        # Frame range of a packed video (multiple episodes in one mp4);
                        # None for clean datasets.
                        "start_frame": item.get("start_frame"),
                        "end_frame": item.get("end_frame"),
                    }
                    continue

                # Mode A: legacy EPIC layout (also accepts "video_path" key)
                video_path = item.get("path") or item.get("video_path")
                if not video_path:
                    continue
                if self.depth_root is None or self.flow_root is None:
                    raise ValueError(
                        "Legacy JSON item with only 'path' requires depth_root and flow_root. "
                        "Either pass them, or use the explicit JSON format with 'rgb'/'depth_npz'/'flow_npz'."
                    )

                scene_rel = _get_sub_dir(video_path)
                scene_id_safe = scene_rel.replace(" ", "_")

                if scene_filter and scene_id_safe not in scene_filter:
                    continue
                if scene_id_safe in self.scene_meta:
                    continue  # Skip duplicates

                # Construct paths (assume they exist - will fail gracefully during loading if not)
                depth_npz = (
                    self.depth_root / scene_rel / "exports" / "mini_npz" / "results.npz"
                )
                flow_npz = self.flow_root / scene_rel / "flows.npz"
                flow_mp4 = self.flow_root / scene_rel / "flow.mp4"

                self.scene_list.append(scene_id_safe)
                self.scene_meta[scene_id_safe] = {
                    "video_path": video_path,
                    "depth_npz": str(depth_npz),
                    "flow_npz": str(flow_npz),
                    "flow_mp4": str(flow_mp4),
                    "original_rel": scene_rel,
                }

        self.scene_list = sorted(self.scene_list)

        # Limit scenes if requested
        if self.max_scenes > 0 and len(self.scene_list) > self.max_scenes:
            logger.info(
                f"Limiting to first {self.max_scenes} scenes (out of {len(self.scene_list)})"
            )
            limited = self.scene_list[: self.max_scenes]
            # Remove meta for excluded scenes
            for sid in self.scene_list[self.max_scenes :]:
                del self.scene_meta[sid]
            self.scene_list = limited

        # Pre-compute frame counts with local cache to avoid repeated network scans
        # Cache key based on depth_root + flow_root + scene list hash
        cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "rynnlam"
            / "frame_counts"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Include source paths and manifest metadata: IDs alone leave stale counts
        # when the same scene is reprocessed or the data root changes.
        # Stream the same JSON representation to retain existing cache keys without
        # materializing a second scene list or a potentially gigabyte-sized string.
        hasher = hashlib.sha256()
        hasher.update(b"[")
        for index, scene_id in enumerate(self.scene_list):
            if index:
                hasher.update(b", ")
            hasher.update(
                json.dumps(
                    (scene_id, self._st_meta.get(scene_id, self.scene_meta[scene_id])),
                    sort_keys=True,
                ).encode()
            )
        hasher.update(b"]")
        scene_hash = hasher.hexdigest()[:20]
        cache_path = (
            cache_dir / f"frame_counts_{scene_hash}_{len(self.scene_list)}.json"
        )

        cached_info = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached_info = json.load(f)
                logger.info(
                    f"Loaded frame count cache: {cache_path} ({len(cached_info)} entries)"
                )
            except Exception:
                cached_info = {}

        self.scene_frame_counts: Dict[str, int] = {}
        valid_scenes = []
        uncached_scenes = []
        new_cache_entries = {}

        # Manifest metadata is already available locally: do not create a Future
        # per scene (millions of objects for a cold combined manifest).
        for scene_id in self.scene_list:
            entry = self._st_meta.get(scene_id, {})
            if all(entry.get(k) is not None for k in ("height", "width", "num_frames")):
                info = {k: entry[k] for k in ("height", "width", "num_frames")}
                info["has_flow_npz"] = True
                if scene_id not in cached_info:
                    new_cache_entries[scene_id] = info
            else:
                info = cached_info.get(scene_id)
            if info is not None:
                meta = self.scene_meta[scene_id]
                meta["height"] = info["height"]
                meta["width"] = info["width"]
                meta["num_frames"] = info["num_frames"]
                meta["has_flow_npz"] = info.get("has_flow_npz", False)
                meta["valid"] = True
                S = self.max_frame_stride
                num_pairs = max(0, (meta["num_frames"] - 1) // S)
                self.scene_frame_counts[scene_id] = num_pairs
                valid_scenes.append(scene_id)
            else:
                uncached_scenes.append(scene_id)

        # Compute uncached scenes (network reads required) — use parallel IO
        if uncached_scenes and self.skip_frame_scan:
            # Skip the slow per-scene network header scan (blocks startup on a
            # throttled NAS when there are ~1M scenes). Placeholders are used;
            # preprocessing decodes each scene anyway, so the real num_frames
            # comes from process_scene. NOTE: pass an explicit --bucket, since
            # auto-bucketing needs native height/width that the scan provides.
            logger.info(
                f"skip_frame_scan=True: NOT scanning {len(uncached_scenes)} "
                f"uncached scenes (placeholders; {len(self.scene_list) - len(uncached_scenes)} cached)"
            )
            for scene_id in uncached_scenes:
                meta = self.scene_meta[scene_id]
                meta.setdefault("height", 0)
                meta.setdefault("width", 0)
                meta.setdefault("num_frames", 0)
                meta.setdefault("has_flow_npz", True)
                meta["valid"] = True
                self.scene_frame_counts[scene_id] = 1
                valid_scenes.append(scene_id)
        elif uncached_scenes:
            workers = int(os.environ.get("LAM_SCAN_THREADS", "64"))
            logger.info(
                f"Scanning {len(uncached_scenes)} uncached scenes with {workers} threads "
                f"({len(self.scene_list) - len(uncached_scenes)} cached)..."
            )

            def _scan_one(scene_id):
                """Read only the raw depth NPZ header for missing frame metadata."""
                m = self.scene_meta[scene_id]
                try:
                    with zipfile.ZipFile(m["depth_npz"], "r") as zf:
                        with zf.open("depth.npy") as f:
                            f.read(6)  # magic
                            ver = struct.unpack("<BB", f.read(2))
                            hlen = (
                                struct.unpack("<H", f.read(2))[0]
                                if ver[0] == 1
                                else struct.unpack("<I", f.read(4))[0]
                            )
                            hdr = ast.literal_eval(
                                f.read(hlen).decode("latin1").strip()
                            )
                            shape = hdr["shape"]  # (num_frames, H, W)
                    num_frames, H, W = int(shape[0]), int(shape[1]), int(shape[2])
                    has_flow_npz = Path(m["flow_npz"]).exists()
                    return scene_id, {
                        "height": H,
                        "width": W,
                        "num_frames": num_frames,
                        "has_flow_npz": has_flow_npz,
                    }
                except Exception as e:
                    return scene_id, None

            failed = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pbar = tqdm(
                    total=len(uncached_scenes), desc="Scanning scenes (parallel)"
                )
                for future in _bounded_scan_futures(
                    pool, _scan_one, uncached_scenes, workers * 2
                ):
                    scene_id, info = future.result()
                    if info is not None:
                        meta = self.scene_meta[scene_id]
                        meta["height"] = info["height"]
                        meta["width"] = info["width"]
                        meta["num_frames"] = info["num_frames"]
                        meta["has_flow_npz"] = info["has_flow_npz"]
                        meta["valid"] = True
                        S = self.max_frame_stride
                        num_pairs = max(0, (info["num_frames"] - 1) // S)
                        self.scene_frame_counts[scene_id] = num_pairs
                        valid_scenes.append(scene_id)
                        new_cache_entries[scene_id] = info
                    else:
                        self.scene_meta[scene_id]["valid"] = False
                        failed += 1
                    pbar.update(1)
                pbar.close()

            if failed > 0:
                logger.warn(f"Skipped {failed} invalid scenes during scanning")

        else:
            logger.info(
                "All frame counts loaded from manifest/cache (0 network reads needed)"
            )

        if new_cache_entries:
            cached_info.update(new_cache_entries)
            try:
                with open(cache_path, "w") as f:
                    json.dump(cached_info, f)
                logger.info(
                    f"Saved frame count cache: {cache_path} ({len(cached_info)} entries)"
                )
            except Exception as e:
                logger.warn(f"Failed to save cache: {e}")

        # Update scene_list to only valid scenes (maintain sorted order)
        valid_set = set(valid_scenes)
        self.scene_list = [s for s in self.scene_list if s in valid_set]
        self.total_frame_pairs = sum(self.scene_frame_counts.values())

        # Build numpy arrays for fast indexing
        self.scene_ids_array = np.array(self.scene_list)
        self.cumulative_counts = np.cumsum(
            [self.scene_frame_counts[s] for s in self.scene_list]
        )

        # Resolution-bucket index for the load-failure fallback: (H, W) -> scene_ids.
        # A substituted sample must match the failed sample's resolution so the
        # batch stays single-resolution.
        self._scenes_by_hw: Dict[Tuple[int, int], List[str]] = {}
        for s in self.scene_list:
            if self.scene_frame_counts.get(s, 0) <= 0:
                continue
            m = self.scene_meta.get(s)
            if not m or "height" not in m or "width" not in m:
                continue
            self._scenes_by_hw.setdefault((m["height"], m["width"]), []).append(s)
        self._fallback_warned = set()

        logger.info(
            f"LAMEpicDataset: {self.total_frame_pairs} non-overlapping chunks from "
            f"{len(self.scene_list)} valid scenes (max_stride={self.max_frame_stride}, "
            f"random stride in [1, {self.max_frame_stride}])"
        )
        npz_count = sum(
            1 for s in self.scene_list if self.scene_meta[s].get("has_flow_npz", False)
        )
        if npz_count > 0:
            logger.info(
                f"  {npz_count}/{len(self.scene_list)} scenes use raw flows.npz (lossless)"
            )
        else:
            logger.info(
                f"  All scenes use flow.mp4 (lossy). Run generate_flows_npz.py to improve quality."
            )

        # Per-worker LRU caches (avoid redundant npz decompression within a batch)
        self._flow_cache = _WorkerLRUCache(max_items=2)
        self._depth_cache = _WorkerLRUCache(max_items=2)
        self._video_cache = _WorkerLRUCache(max_items=1)
        self._st_handle_cache = _WorkerLRUCache(
            max_items=4000
        )  # open safe_open handles for slice reads

    def __len__(self) -> int:
        return self.total_frame_pairs

    def _idx_to_scene_frame(self, idx: int) -> Tuple[str, int]:
        """Convert linear index to (scene_id, chunk_start_frame) using binary search."""
        scene_idx = np.searchsorted(self.cumulative_counts, idx + 1)
        scene_id = self.scene_ids_array[scene_idx]

        start_idx = (
            self.cumulative_counts[scene_idx] - self.scene_frame_counts[scene_id]
        )
        chunk_idx = idx - start_idx
        # Non-overlapping: chunk start = chunk_idx * max_frame_stride
        chunk_start = int(chunk_idx) * self.max_frame_stride

        return scene_id, chunk_start

    def get_scene(self, scene_id: str) -> List[Dict]:
        """Get all frame pairs for a given scene (for evaluation)."""
        # Safetensors-backed scenes (combined dataset): iterate frame pairs via
        # the standard on-demand loader, which handles the stride logic.
        if scene_id in self._st_meta:
            st_entry = self._st_meta[scene_id]
            num_frames = st_entry["num_frames"]
            stride = self.max_frame_stride
            samples = []
            for chunk_start in range(0, num_frames - stride, stride):
                sample = self._load_from_safetensors(scene_id, chunk_start)
                sample["scene"] = scene_id
                sample["frame_idx"] = chunk_start
                samples.append(sample)
            return samples

        if scene_id not in self.scene_meta:
            raise ValueError(
                f"Scene '{scene_id}' not found. Available: {self.scene_list[:5]}..."
            )

        meta = self.scene_meta[scene_id]
        H, W = meta["height"], meta["width"]

        # Load all frames
        cap = cv2.VideoCapture(meta["video_path"])
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = (
                cv2.resize(frame, (W, H), interpolation=cv2.INTER_CUBIC).astype(
                    np.float32
                )
                / 255.0
            )
            all_frames.append(torch.from_numpy(frame))
        cap.release()

        # Load depth/camera
        npz_data = np.load(meta["depth_npz"])
        extrinsics_all = npz_data["extrinsics"]
        intrinsics_all = npz_data["intrinsics"]
        depth_all = npz_data["depth"]
        T = extrinsics_all.shape[0]
        npz_data.close()

        ext_4x4 = np.zeros((T, 4, 4), dtype=np.float32)
        ext_4x4[:, :3, :] = extrinsics_all
        ext_4x4[:, 3, 3] = 1.0

        # Load flow
        use_npz = meta.get("has_flow_npz", False)
        if use_npz:
            npz_flow_data = np.load(meta["flow_npz"])
            optical_flows = npz_flow_data["flows"]  # [N-1, H, W, 2]
            npz_flow_data.close()
        else:
            optical_flows = _decode_flow_video(meta["flow_mp4"], self.flow_max_radius)
        stride = self.max_frame_stride

        # Non-overlapping chunks with fixed max stride for eval
        samples = []
        for chunk_start in range(0, T - stride, stride):
            idx2 = chunk_start + stride
            images = torch.stack([all_frames[chunk_start], all_frames[idx2]], dim=0)
            extrinsics = torch.from_numpy(
                np.stack([ext_4x4[chunk_start], ext_4x4[idx2]], axis=0)
            )
            intrinsics = torch.from_numpy(
                np.stack(
                    [intrinsics_all[chunk_start], intrinsics_all[idx2]], axis=0
                ).astype(np.float32)
            )
            depths = torch.from_numpy(
                np.stack([depth_all[chunk_start], depth_all[idx2]], axis=0).astype(
                    np.float32
                )
            )

            # Chain flows for stride > 1
            if stride == 1:
                optical_flow = torch.from_numpy(optical_flows[chunk_start])
            else:
                optical_flow = torch.from_numpy(
                    _chain_flow_sequence(optical_flows[chunk_start:idx2])
                )

            flow, mask, _, _ = self.compute_flow_t(
                extrinsics, intrinsics, depths, optical_flow
            )

            samples.append(
                {
                    "images": images,
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "flow": flow,
                    "mask": mask.unsqueeze(-1),
                    "depths": depths,
                    "scene": scene_id,
                    "frame_idx": chunk_start,
                }
            )

        return samples

    def __getitem__(self, idx: int) -> Dict:
        scene_id, frame_idx = self._idx_to_scene_frame(idx)
        try:
            return self._load_on_demand(scene_id, frame_idx)
        except Exception as e:
            return self._resample_fallback(scene_id, frame_idx, e)

    def _resample_fallback(
        self, scene_id: str, frame_idx: int, orig_err: Exception
    ) -> Dict:
        """Substitute an unreadable sample with a random same-resolution one.

        A handful of scenes out of hundreds of thousands are corrupt (truncated
        safetensors, num_frames overstated past the actual data, missing files).
        Rather than crash the whole DDP job, replace the bad sample with a random
        sample from the same (H, W) bucket so the batch stays single-resolution.
        """
        meta = self.scene_meta.get(scene_id, {})
        hw = (meta.get("height"), meta.get("width"))
        candidates = self._scenes_by_hw.get(hw, [])
        if scene_id not in self._fallback_warned:
            self._fallback_warned.add(scene_id)
            logger.warn(
                f"[data] unreadable sample scene='{scene_id}' frame={frame_idx} "
                f"({orig_err!r}); substituting a same-resolution sample"
            )
        attempts = 8
        last_err = orig_err
        for _ in range(attempts):
            if not candidates:
                break
            alt = candidates[random.randrange(len(candidates))]
            n_chunks = self.scene_frame_counts.get(alt, 0)
            if n_chunks <= 0 or alt == scene_id:
                continue
            alt_frame = random.randrange(n_chunks) * self.max_frame_stride
            try:
                return self._load_on_demand(alt, alt_frame)
            except Exception as e2:
                last_err = e2
                continue
        raise RuntimeError(
            f"Failed to load sample and {attempts} fallbacks failed. "
            f"scene='{scene_id}' frame={frame_idx}: {orig_err!r}; "
            f"last fallback error: {last_err!r}"
        ) from orig_err

    # =====================================================================
    # On-demand loading
    # =====================================================================

    def _load_on_demand(self, scene_id: str, frame_idx: int) -> Dict:
        # ---- Fast path: precomputed safetensors ----
        if scene_id in self._st_meta:
            return self._load_from_safetensors(scene_id, frame_idx)

        meta = self.scene_meta[scene_id]
        native_H, native_W = meta["height"], meta["width"]
        num_frames = meta["num_frames"]

        # Use target resolution if set, otherwise use native
        H, W = self.target_hw if self.target_hw else (native_H, native_W)

        # Random stride in [min_frame_stride, max_sample_stride], clamped to not exceed video bounds
        min_s = getattr(self, "min_frame_stride", 1)
        max_s = min(self.max_sample_stride, num_frames - 1 - frame_idx)
        if max_s < min_s:
            max_s = min_s
        stride = np.random.randint(min_s, max_s + 1)
        frame_idx2 = frame_idx + stride

        # Frames
        images = self._load_image_pair(meta["video_path"], frame_idx, frame_idx2, H, W)

        # Depth/camera (pass target H,W for resizing if needed)
        extrinsics, intrinsics, depths = self._load_camera_and_depth(
            meta["depth_npz"], frame_idx, frame_idx2, H, W
        )

        # Flow: prefer raw NPZ, fallback to flow.mp4
        if meta.get("has_flow_npz", False):
            optical_flow = self._load_optical_flow_npz(
                meta["flow_npz"], frame_idx, stride, H, W
            )
        else:
            optical_flow = self._load_optical_flow_chained(
                meta["flow_mp4"], frame_idx, stride, H, W
            )

        # Compute GT 3D flow (fast numpy path for CPU dataloader workers)
        flow, mask = self._compute_flow_t_numpy(
            extrinsics.numpy(), intrinsics.numpy(), depths.numpy(), optical_flow.numpy()
        )

        return {
            "images": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "flow": torch.from_numpy(flow),
            "mask": torch.from_numpy(mask[..., None]),
            "depths": depths,
            "scene": scene_id,
            "frame_idx": frame_idx,
            "domain_id": torch.tensor(
                domain_id_from_scene_id(scene_id), dtype=torch.long
            ),
            "view_id": torch.tensor(view_id_from_scene_id(scene_id), dtype=torch.long),
        }

    def _load_from_safetensors(self, scene_id: str, frame_idx: int) -> Dict:
        """Load a sample using lazy slice reads (safe_open.get_slice) to avoid loading full files."""
        st_entry = self._st_meta[scene_id]
        st_file = st_entry["file"]
        num_frames = st_entry["num_frames"]
        # Tensor prefix = the file's own stem (self-describing). Legacy manifests
        # remapped scene_ids (added dataset prefix, normalized role names like
        # cam_high -> head) WITHOUT rewriting the tensor keys inside the files,
        # so a scene_id-derived prefix KeyErrors on every legacy scene.
        # Sharded entries (many scenes per file) keep the scene-derived key.
        safe_key = st_entry.get("tensor_prefix") or (
            scene_id.replace("/", "__") if "shard" in st_entry else Path(st_file).stem
        )

        min_s = self.min_frame_stride
        max_s = min(self.max_sample_stride, num_frames - 1 - frame_idx)
        if max_s < min_s:
            max_s = min_s
        stride = np.random.randint(min_s, max_s + 1)
        frame_idx2 = frame_idx + stride

        st = self._st_handle_cache.get(st_file)
        if st is None:
            st = safe_open(st_file, framework="np", device="cpu")
            self._st_handle_cache.put(st_file, st)

        img1 = (
            st.get_slice(f"{safe_key}__rgb")[frame_idx].astype(np.float32) / 255.0
        )
        img2 = (
            st.get_slice(f"{safe_key}__rgb")[frame_idx2].astype(np.float32) / 255.0
        )
        images = torch.from_numpy(np.stack([img1, img2], axis=0))
        # Stale-metadata guard (raises -> fallback).
        exp_h, exp_w = int(st_entry.get("height", 0)), int(st_entry.get("width", 0))
        if exp_h > 0 and (images.shape[1], images.shape[2]) != (exp_h, exp_w):
            raise ValueError(
                f"manifest says {exp_h}x{exp_w} but file content is "
                f"{images.shape[1]}x{images.shape[2]} (scene={scene_id})"
            )

        mid_image = None
        if self.use_additivity:
            mid_off = int(np.random.randint(1, stride)) if stride >= 2 else 1
            frame_idx_mid = frame_idx + mid_off
            img_mid = (
                st.get_slice(f"{safe_key}__rgb")[frame_idx_mid].astype(np.float32)
                / 255.0
            )
            mid_image = torch.from_numpy(img_mid).unsqueeze(0)  # [1, H, W, 3]

        d1 = st.get_slice(f"{safe_key}__depth")[frame_idx].astype(np.float32)
        d2 = st.get_slice(f"{safe_key}__depth")[frame_idx2].astype(np.float32)
        depths = torch.from_numpy(np.stack([d1, d2], axis=0))

        ext = torch.from_numpy(
            np.stack(
                [
                    st.get_slice(f"{safe_key}__extrinsics")[frame_idx],
                    st.get_slice(f"{safe_key}__extrinsics")[frame_idx2],
                ],
                axis=0,
            )
        )
        intr = torch.from_numpy(
            np.stack(
                [
                    st.get_slice(f"{safe_key}__intrinsics")[frame_idx],
                    st.get_slice(f"{safe_key}__intrinsics")[frame_idx2],
                ],
                axis=0,
            )
        )

        # Check flow shape to determine loading strategy
        flow_shape = st.get_slice(f"{safe_key}__flow").get_shape()
        if frame_idx < 0 or frame_idx + stride > flow_shape[0]:
            raise ValueError(
                f"Incomplete flow interval [{frame_idx}, {frame_idx + stride}) "
                f"for {flow_shape[0]} flows (scene={scene_id})"
            )
        is_3d_flow = len(flow_shape) == 4 and flow_shape[-1] == 3

        if is_3d_flow:
            # Precomputed 3D flow [T-1, H, W, 3]: accumulate over stride
            flow_seq = st.get_slice(f"{safe_key}__flow")[
                frame_idx : frame_idx + stride
            ].astype(np.float32)
            optical_flow = flow_seq.sum(axis=0)  # [H, W, 3]
        elif stride == 1:
            optical_flow = st.get_slice(f"{safe_key}__flow")[frame_idx].astype(
                np.float32
            )
        else:
            flow_seq = st.get_slice(f"{safe_key}__flow")[
                frame_idx : frame_idx + stride
            ].astype(np.float32)
            optical_flow = _chain_flow_sequence(flow_seq)

        # Check if flow is precomputed 3D flow (SynthVerse: [H,W,3]) vs 2D optical flow ([H,W,2])
        if optical_flow.shape[-1] == 3:
            # Precomputed 3D flow — use directly
            flow = optical_flow[None].astype(np.float32)  # [1, H, W, 3]
            # Build mask from stored mask or depth validity
            d1, d2 = depths.numpy()[0], depths.numpy()[1]
            mask = ((d1 > 0) & np.isfinite(d1) & (d2 > 0) & np.isfinite(d2)).astype(
                np.float32
            )
            # Also zero out flow where mask is invalid
            flow[0][~(mask > 0.5)] = 0.0
            mask = mask[None]  # [1, H, W]
        else:
            # Standard 2D optical flow → compute 3D flow
            flow, mask = self._compute_flow_t_numpy(
                ext.numpy(), intr.numpy(), depths.numpy(), optical_flow
            )

        result = {
            "images": images,
            "extrinsics": ext,
            "intrinsics": intr,
            "flow": torch.from_numpy(flow),
            "mask": torch.from_numpy(mask[..., None]),
            "depths": depths,
            "scene": scene_id,
            "frame_idx": frame_idx,
            "dataset": st_entry.get("dataset", "unknown"),
            "role": st_entry.get("role") or role_from_scene_id(scene_id),
        }
        if mid_image is not None:
            result["mid_images"] = mid_image  # [1, H, W, 3]
        return result

    def _compute_flow_t_numpy(self, extrinsics, intrinsics, depths, optical_flow):
        """Fast numpy implementation of compute_flow_t for CPU dataloader workers."""
        H, W = depths.shape[-2:]
        d1, d2 = depths[0], depths[1]
        K1, K2 = intrinsics[0], intrinsics[1]
        E1, E2 = extrinsics[0], extrinsics[1]

        ys, xs = np.mgrid[:H, :W].astype(np.float32)

        # Backproject
        pts1 = np.stack(
            [(xs - K1[0, 2]) * d1 / K1[0, 0], (ys - K1[1, 2]) * d1 / K1[1, 1], d1],
            axis=-1,
        )
        pts2 = np.stack(
            [(xs - K2[0, 2]) * d2 / K2[0, 0], (ys - K2[1, 2]) * d2 / K2[1, 1], d2],
            axis=-1,
        )

        # Transform pts2 from cam_t+1 to cam_t
        T = E1 @ np.linalg.inv(E2)
        ones = np.ones((H, W, 1), dtype=np.float32)
        pts2_h = np.concatenate([pts2, ones], axis=-1).reshape(-1, 4)
        pts2_in_t = (pts2_h @ T.T)[:, :3].reshape(H, W, 3)

        # Optical flow correspondence
        flow_u = optical_flow[..., 0]
        flow_v = optical_flow[..., 1]
        tgt_x = xs + flow_u
        tgt_y = ys + flow_v

        in_bounds = (tgt_x >= 0) & (tgt_x <= W - 1) & (tgt_y >= 0) & (tgt_y <= H - 1)
        txr = np.round(tgt_x).astype(np.int32).clip(0, W - 1)
        tyr = np.round(tgt_y).astype(np.int32).clip(0, H - 1)

        pts2_corr = pts2_in_t[tyr, txr]
        flow_3d = pts2_corr - pts1

        # Depth gradient masks
        def depth_grad_mask(dp):
            gx = np.zeros_like(dp)
            gy = np.zeros_like(dp)
            gx[:, 1:-1] = 0.5 * (dp[:, 2:] - dp[:, :-2])
            gy[1:-1, :] = 0.5 * (dp[2:, :] - dp[:-2, :])
            return (
                (np.sqrt(gx**2 + gy**2) < self.depth_grad_threshold)
                & np.isfinite(dp)
                & (dp > 0)
            )

        m1 = depth_grad_mask(d1)
        m2c = depth_grad_mask(d2)[tyr, txr]
        valid = in_bounds & m1 & m2c & np.isfinite(flow_3d).all(axis=-1)
        flow_3d[~valid] = 0.0

        return flow_3d[None].astype(np.float32), valid[None].astype(np.float32)

    def _load_image_pair(
        self, video_path: str, idx1: int, idx2: int, H: int, W: int
    ) -> torch.Tensor:
        cache_key = f"{video_path}_{H}_{W}"
        cached = self._video_cache.get(cache_key)
        if cached is None:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_CUBIC)
                frames.append(frame)
            cap.release()
            if not frames:
                raise RuntimeError(f"Failed to read any frames from {video_path}")
            cached = frames
            self._video_cache.put(cache_key, cached)

        img_t = cached[idx1].astype(np.float32) / 255.0
        img_t1 = cached[idx2].astype(np.float32) / 255.0

        return torch.from_numpy(np.stack([img_t, img_t1], axis=0))

    def _load_camera_and_depth(
        self, npz_path: str, idx1: int, idx2: int, target_H: int = 0, target_W: int = 0
    ):
        cached = self._depth_cache.get(npz_path)
        if cached is None:
            npz_data = np.load(npz_path, allow_pickle=False)
            extrinsics_all = npz_data["extrinsics"]
            intrinsics_all = npz_data["intrinsics"]
            depth_all = npz_data["depth"]
            npz_data.close()
            N = extrinsics_all.shape[0]
            ext_4x4 = np.zeros((N, 4, 4), dtype=np.float32)
            ext_4x4[:, :3, :] = extrinsics_all
            ext_4x4[:, 3, 3] = 1.0
            cached = (ext_4x4, intrinsics_all, depth_all)
            self._depth_cache.put(npz_path, cached)

        ext_4x4, intrinsics_all, depth_all = cached

        extrinsics = torch.from_numpy(np.stack([ext_4x4[idx1], ext_4x4[idx2]], axis=0))
        intrinsics = np.stack(
            [intrinsics_all[idx1], intrinsics_all[idx2]], axis=0
        ).astype(np.float32)

        d1 = depth_all[idx1].astype(np.float32)
        d2 = depth_all[idx2].astype(np.float32)
        native_H, native_W = d1.shape[:2]

        # Resize depth and adjust intrinsics if target resolution differs
        if (
            target_H > 0
            and target_W > 0
            and (native_H != target_H or native_W != target_W)
        ):
            scale_x = target_W / native_W
            scale_y = target_H / native_H
            d1 = cv2.resize(d1, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
            d2 = cv2.resize(d2, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
            for i in range(2):
                intrinsics[i, 0, 0] *= scale_x  # fx
                intrinsics[i, 0, 2] *= scale_x  # cx
                intrinsics[i, 1, 1] *= scale_y  # fy
                intrinsics[i, 1, 2] *= scale_y  # cy

        depths = torch.from_numpy(np.stack([d1, d2], axis=0))
        intrinsics = torch.from_numpy(intrinsics)

        return extrinsics, intrinsics, depths

    def _load_optical_flow(
        self, flow_mp4_path: str, frame_idx: int, target_H: int = 0, target_W: int = 0
    ) -> torch.Tensor:
        cap = cv2.VideoCapture(flow_mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(
                f"Failed to read flow frame {frame_idx} from {flow_mp4_path}"
            )
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flow = _decode_flow_frame(frame_rgb, flow_max_radius=self.flow_max_radius)

        # Resize flow to match depth/image resolution if needed
        fH, fW = flow.shape[:2]
        if target_H > 0 and target_W > 0 and (fH != target_H or fW != target_W):
            # Scale flow values proportionally to the resolution change
            scale_x = target_W / fW
            scale_y = target_H / fH
            flow_resized = cv2.resize(
                flow, (target_W, target_H), interpolation=cv2.INTER_LINEAR
            )
            flow_resized[..., 0] *= scale_x
            flow_resized[..., 1] *= scale_y
            flow = flow_resized

        return torch.from_numpy(flow)

    def _load_optical_flow_chained(
        self,
        flow_mp4_path: str,
        frame_idx: int,
        stride: int,
        target_H: int = 0,
        target_W: int = 0,
    ) -> torch.Tensor:
        """Load and chain `stride` consecutive flow frames to get flow from frame_idx to frame_idx+stride.

        Flow chaining: for each pixel at frame i, follow the flow through intermediate frames
        to get the total displacement to frame i+stride.
        """
        if stride == 1:
            return self._load_optical_flow(flow_mp4_path, frame_idx, target_H, target_W)

        # Read stride consecutive flow frames (flow[i->i+1], flow[i+1->i+2], ..., flow[i+k-1->i+k])
        cap = cv2.VideoCapture(flow_mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        flows = []
        for _ in range(stride):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flow = _decode_flow_frame(frame_rgb, flow_max_radius=self.flow_max_radius)
            flows.append(flow)
        cap.release()

        if len(flows) == 0:
            raise RuntimeError(
                f"Failed to read any flow frame at {frame_idx} from {flow_mp4_path}"
            )

        # Chain flows: accumulate displacement through intermediate frames
        fH, fW = flows[0].shape[:2]
        # Start with identity grid (pixel coordinates)
        ys, xs = np.mgrid[:fH, :fW].astype(np.float32)
        # Current position of each pixel as it's warped forward
        cur_x = xs.copy()
        cur_y = ys.copy()

        for flow_i in flows:
            # Bilinear sample flow_i at (cur_x, cur_y)
            # Clamp to valid range
            sx = np.clip(cur_x, 0, fW - 1)
            sy = np.clip(cur_y, 0, fH - 1)

            # Integer and fractional parts for bilinear interpolation
            x0 = np.floor(sx).astype(np.int32)
            y0 = np.floor(sy).astype(np.int32)
            x1 = np.minimum(x0 + 1, fW - 1)
            y1 = np.minimum(y0 + 1, fH - 1)
            wx = sx - x0
            wy = sy - y0

            # Vectorized bilinear interpolation for both channels
            w00 = ((1 - wx) * (1 - wy))[..., None]
            w01 = (wx * (1 - wy))[..., None]
            w10 = ((1 - wx) * wy)[..., None]
            w11 = (wx * wy)[..., None]

            interp = (
                flow_i[y0, x0] * w00
                + flow_i[y0, x1] * w01
                + flow_i[y1, x0] * w10
                + flow_i[y1, x1] * w11
            )

            cur_x += interp[..., 0]
            cur_y += interp[..., 1]

        # Total flow = final position - original position
        chained_flow = np.stack([cur_x - xs, cur_y - ys], axis=-1).astype(np.float32)

        # Resize if needed
        if target_H > 0 and target_W > 0 and (fH != target_H or fW != target_W):
            scale_x = target_W / fW
            scale_y = target_H / fH
            chained_flow = cv2.resize(
                chained_flow, (target_W, target_H), interpolation=cv2.INTER_LINEAR
            )
            chained_flow[..., 0] *= scale_x
            chained_flow[..., 1] *= scale_y

        return torch.from_numpy(chained_flow)

    def _load_optical_flow_npz(
        self,
        npz_path: str,
        frame_idx: int,
        stride: int,
        target_H: int = 0,
        target_W: int = 0,
    ) -> torch.Tensor:
        """Load raw float32 optical flow from flows.npz and chain if stride > 1.

        flows.npz contains 'flows' array of shape [N-1, H, W, 2] where
        flows[i] = optical flow from frame i to frame i+1 (raw model output, no compression).

        Uses per-worker LRU cache to avoid repeated decompression for same-scene samples.
        """
        cached = self._flow_cache.get(npz_path)
        if cached is None:
            npz_data = np.load(npz_path, allow_pickle=False)
            cached = npz_data["flows"]  # keep in native dtype (float16) to save memory
            npz_data.close()
            self._flow_cache.put(npz_path, cached)

        if stride == 1:
            flow = cached[frame_idx].astype(np.float32)
        else:
            flow_seq = cached[frame_idx : frame_idx + stride].astype(np.float32)
            flow = _chain_flow_sequence(flow_seq)

        fH, fW = flow.shape[:2]
        if target_H > 0 and target_W > 0 and (fH != target_H or fW != target_W):
            scale_x = target_W / fW
            scale_y = target_H / fH
            flow = cv2.resize(
                flow, (target_W, target_H), interpolation=cv2.INTER_LINEAR
            )
            flow[..., 0] *= scale_x
            flow[..., 1] *= scale_y

        return torch.from_numpy(flow)

    # =====================================================================
    # GT 3D flow computation
    # =====================================================================

    def compute_flow_t(self, extrinsics, intrinsics, depths, optical_flow):
        device = depths.device
        dtype = depths.dtype
        H, W = depths.shape[-2:]

        depth1, depth2 = depths[0], depths[1]
        K1, K2 = intrinsics[0], intrinsics[1]
        E1, E2 = extrinsics[0], extrinsics[1]

        def make_pixel_grid(H, W, device, dtype):
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij",
            )
            return xs, ys

        def backproject_to_camera(depth, intrinsic):
            xs, ys = make_pixel_grid(H, W, depth.device, depth.dtype)
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            z = depth
            x = (xs - cx) * z / fx
            y = (ys - cy) * z / fy
            return torch.stack([x, y, z], dim=-1)

        def transform_points_cam_to_cam(points_src_cam, T_dst_src):
            ones = torch.ones(
                (H, W, 1), device=points_src_cam.device, dtype=points_src_cam.dtype
            )
            points_src_h = torch.cat([points_src_cam, ones], dim=-1)
            points_dst_h = points_src_h.view(-1, 4) @ T_dst_src.T
            return points_dst_h[:, :3].view(H, W, 3)

        def depth_gradient_mask(depth, threshold):
            gx = torch.zeros_like(depth)
            gy = torch.zeros_like(depth)
            gx[:, 1:-1] = 0.5 * (depth[:, 2:] - depth[:, :-2])
            gy[1:-1, :] = 0.5 * (depth[2:, :] - depth[:-2, :])
            grad_mag = torch.sqrt(gx**2 + gy**2)
            mask = grad_mag < threshold
            mask = mask & torch.isfinite(depth) & (depth > 0)
            return mask

        depth_mask1 = depth_gradient_mask(depth1, self.depth_grad_threshold)
        depth_mask2 = depth_gradient_mask(depth2, self.depth_grad_threshold)

        pts1_cam = backproject_to_camera(depth1, K1)
        pts2_cam = backproject_to_camera(depth2, K2)

        T_t_from_t1 = E1 @ torch.linalg.inv(E2)
        pts2_in_t_cam = transform_points_cam_to_cam(pts2_cam, T_t_from_t1)

        flow_u = optical_flow[..., 0]
        flow_v = optical_flow[..., 1]
        xs, ys = make_pixel_grid(H, W, device, dtype)
        tgt_x = xs + flow_u
        tgt_y = ys + flow_v

        in_bounds = (tgt_x >= 0) & (tgt_x <= W - 1) & (tgt_y >= 0) & (tgt_y <= H - 1)
        tgt_x_round = torch.round(tgt_x).long().clamp(0, W - 1)
        tgt_y_round = torch.round(tgt_y).long().clamp(0, H - 1)

        pts2_corr_in_t_cam = pts2_in_t_cam[tgt_y_round, tgt_x_round]
        depth_mask2_corr = depth_mask2[tgt_y_round, tgt_x_round]

        flow_3d = pts2_corr_in_t_cam - pts1_cam
        valid_mask = (
            in_bounds
            & depth_mask1
            & depth_mask2_corr
            & torch.isfinite(flow_3d).all(dim=-1)
        )
        flow_3d = torch.where(
            valid_mask.unsqueeze(-1), flow_3d, torch.zeros_like(flow_3d)
        )

        return flow_3d.unsqueeze(0), valid_mask.unsqueeze(0), depth_mask1, depth_mask2


def lam_collate_fn(batch: List[Dict]) -> Dict:
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values, dim=0)
        else:
            result[key] = values
    return result


epic_collate_fn = lam_collate_fn


class BucketedDistributedBatchSampler(Sampler):
    """Distributed batch sampler that never mixes resolutions inside a batch.

    Preprocessed datasets are bucketed by aspect ratio, so
    one `safetensors_root` can hold e.g. RoboMIND's 238x322 *and* 210x364 scenes.
    `lam_collate_fn` stacks tensors, which fails outright on mixed resolutions —
    hence batches must be built per bucket rather than from a global shuffle.

    Every rank is handed the *same number* of batches. An unequal count would let
    one rank finish early and leave the others blocking in a DDP collective until
    the NCCL timeout — the same failure mode as the per-rank NaN skip bug fixed in
    train_motion_8gpu.py. Pass this as DataLoader(batch_sampler=...).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.bucket_to_indices = self._group_by_bucket()
        sizes = {b: len(v) for b, v in self.bucket_to_indices.items()}
        logger.info(
            f"BucketedDistributedBatchSampler: {len(sizes)} bucket(s) "
            f"{sizes}, batch_size={batch_size}, replicas={num_replicas}"
        )

        # Batch count is deterministic (it does not depend on the shuffle), so it
        # can be computed once for __len__.
        self._num_batches_per_rank = self._count_batches()

    def _group_by_bucket(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Group compact indices in scene-sized spans, not Python pair objects."""
        st_meta = getattr(self.dataset, "_st_meta", None) or {}
        if not st_meta:
            return {(0, 0): np.arange(len(self.dataset), dtype=np.int64)}

        def key(scene_id):
            entry = st_meta.get(scene_id, {})
            return (int(entry.get("height", 0) or 0), int(entry.get("width", 0) or 0))

        return _compact_index_groups(self.dataset, key)

    def _count_batches(self) -> int:
        total = 0
        for indices in self.bucket_to_indices.values():
            n = len(indices)
            total += (
                n // self.batch_size if self.drop_last else -(-n // self.batch_size)
            )
        # Trim so every rank gets the same number of batches.
        return total // self.num_replicas

    def _build_batches(self) -> List[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        batches: List[List[int]] = []
        for _, indices in sorted(self.bucket_to_indices.items()):
            order = (
                torch.randperm(len(indices), generator=g).numpy()
                if self.shuffle
                else np.arange(len(indices))
            )
            shuffled = indices[order]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            # Interleave buckets so a rank does not spend a whole epoch on one
            # resolution (which would correlate resolution with training step).
            batches = [
                batches[i] for i in torch.randperm(len(batches), generator=g).tolist()
            ]
        return batches

    def __iter__(self):
        batches = self._build_batches()
        mine = batches[self.rank :: self.num_replicas]
        # Every rank must run exactly the same number of steps.
        return (batch.tolist() for batch in mine[: self._num_batches_per_rank])

    def __len__(self) -> int:
        return self._num_batches_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


_ROLE_ALIAS = {
    # Droid raw camera names (kept un-normalized in scene_ids)
    "exterior_image_1_left": "global",
    "exterior_image_2_left": "side",
    "wrist_image_left": "wrist_left",
}


def role_from_scene_id(scene_id: str) -> str:
    """View role = last path segment of the scene_id, with Droid aliases folded."""
    role = scene_id.rsplit("/", 1)[-1]
    return _ROLE_ALIAS.get(role, role)


# --- Domain family + view vocabularies for the conditioned latent encoder (D1) ---
# Four domain families, keyed on the dataset prefix of a scene_id: real robot, simulation,
# first-person human video, and everything else. Index order is fixed so checkpoints are
# reproducible; an unlisted dataset maps to "other".
DOMAIN_CLASSES = ["real", "sim", "human", "other"]

_DATASET_TO_DOMAIN = {
    # real robot
    "robomind": "real",
    "robocoin": "real",
    "agibot-beta": "real",
    "rdt-1b": "real",
    "galaxea": "real",
    "droid": "real",
    "interndata-a1": "real",
    "agibotworld2026": "real",
    "openx-embodiment": "real",
    "rovid-x": "real",
    "molmoact2": "real",
    "robomind2.0": "real",
    "table30": "real",
    "table30v2": "real",
    "bridgev2": "real",
    "dreamdojo-eval": "real",
    "wmbench": "real",
    "gr1_robot": "real",
    # sim
    "behavior-1k": "sim",
    "robotwin2.0": "sim",
    "robocasa": "sim",
    "libero_plus": "sim",
    "robodojo": "sim",
    "vlabench": "sim",
    "robotwin_unified": "sim",
    # first-person human ego
    "egovid": "human",
    "howto100m": "human",
    "epic_kitchens": "human",
    "egodex": "human",
    "egoverse": "human",
    "sthsthv2": "human",
}

VIEW_CLASSES = ["head", "wrist_left", "wrist_right", "global", "side", "other"]


def domain_id_from_scene_id(scene_id: str) -> int:
    """Domain family index from the dataset prefix (first path segment)."""
    dataset = scene_id.split("/", 1)[0].lower()
    family = _DATASET_TO_DOMAIN.get(dataset, "other")
    return DOMAIN_CLASSES.index(family)


def view_id_from_scene_id(scene_id: str) -> int:
    """View-role index from the role helper (head/wrist/global/side/other)."""
    role = role_from_scene_id(scene_id)
    return (
        VIEW_CLASSES.index(role)
        if role in VIEW_CLASSES
        else VIEW_CLASSES.index("other")
    )


def _compact_index_groups(dataset, key_for_scene):
    """Preserve index/group order while allocating only one int64 array per group."""

    def spans():
        if hasattr(dataset, "scene_list") and hasattr(dataset, "scene_frame_counts"):
            start = 0
            for scene_id in dataset.scene_list:
                count = dataset.scene_frame_counts[scene_id]
                if count:
                    yield key_for_scene(scene_id), start, count
                start += count
        else:
            # Compatibility for arbitrary datasets without scene span metadata.
            for idx in range(len(dataset)):
                scene_id, _ = dataset._idx_to_scene_frame(idx)
                yield key_for_scene(scene_id), idx, 1

    sizes = {}
    for key, _, count in spans():
        sizes[key] = sizes.get(key, 0) + count
    groups = {key: np.empty(count, dtype=np.int64) for key, count in sizes.items()}
    offsets = dict.fromkeys(groups, 0)
    for key, start, count in spans():
        offset = offsets[key]
        groups[key][offset : offset + count] = np.arange(
            start, start + count, dtype=np.int64
        )
        offsets[key] += count
    return groups


class WeightedMultiDatasetBatchSampler(Sampler):
    """Distributed batch sampler for several merged datasets with per-dataset rates.

    The combined manifest tags every scene with a
    "dataset" field. Left to natural sampling, the biggest datasets (egodex /
    robomind) would dominate every batch. This sampler instead draws a per-dataset
    sample budget proportional to a configurable weight, so small datasets can be
    over-represented (drawn with replacement) and large ones under-represented.

    Per-dataset weight = size ** (1/temperature) * multiplier, where
      * temperature > 1 rebalances toward smaller datasets (1.0 = proportional),
      * multiplier is an optional explicit per-dataset override (e.g. boost a tiny
        dataset further).
    Batches never mix resolutions (same constraint as BucketedDistributedBatchSampler),
    and every rank is handed the same number of batches to keep DDP collectives in
    sync. Pass as DataLoader(batch_sampler=...).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        temperature: float = 1.0,
        dataset_multipliers: Optional[Dict[str, float]] = None,
        role_multipliers: Optional[Dict[str, float]] = None,
        total_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.temperature = float(temperature)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.multipliers = dataset_multipliers or {}
        # Optional per-view-role multiplier applied WITHIN each dataset's budget
        # (e.g. boost wrist/global/side over head on robot datasets). Roles are
        # parsed from the scene_id suffix; unrecognized suffixes -> 1.0.
        self.role_multipliers = role_multipliers or {}
        for mapping in (self.multipliers, self.role_multipliers):
            for key, value in mapping.items():
                if not np.isfinite(float(value)) or float(value) < 0:
                    raise ValueError(
                        f"Multiplier for {key!r} must be finite and nonnegative"
                    )
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Store each pair index once; dataset-level weighting needs counts only.
        self.group_to_indices: Dict[Tuple[str, Tuple[int, int], str], np.ndarray] = {}
        self.dataset_sizes: Dict[str, int] = {}
        self._group()

        self.total_samples = total_samples or len(self.dataset)
        self.weights = self._compute_weights()
        self._per_dataset_n = self._per_dataset_budget()
        self._num_batches_per_rank = self._count_batches()

        if True:
            wsum = sum(self.weights.values()) or 1.0
            share = {d: f"{self.weights[d]/wsum*100:.1f}%" for d in self.weights}
            logger.info(
                f"WeightedMultiDatasetBatchSampler: T={self.temperature} "
                f"datasets={self.dataset_sizes}"
            )
            logger.info(f"  per-dataset sample share: {share}")
            if self.role_multipliers:
                role_w: Dict[str, float] = {}
                for (_ds, _hw, role), idxs in self.group_to_indices.items():
                    if role not in self.role_multipliers and role != "head":
                        role = "other"  # e.g. numeric chunk ids on ego/video sets
                    role_w[role] = role_w.get(role, 0.0) + len(idxs) * float(
                        self.role_multipliers.get(role, 1.0)
                    )
                rw_sum = sum(role_w.values()) or 1.0
                logger.info(
                    f"  role multipliers {self.role_multipliers} -> effective role share: "
                    f"{ {r: f'{w / rw_sum * 100:.1f}%' for r, w in sorted(role_w.items())} }"
                )

    def _group(self):
        st_meta = getattr(self.dataset, "_st_meta", None) or {}

        def key(scene_id):
            entry = st_meta.get(scene_id, {})
            ds = entry.get("dataset", "unknown")
            hw = (int(entry.get("height", 0) or 0), int(entry.get("width", 0) or 0))
            role = entry.get("role") or role_from_scene_id(scene_id)
            return ds, hw, _ROLE_ALIAS.get(role, role)

        self.group_to_indices = _compact_index_groups(self.dataset, key)
        for (ds, _, _), indices in self.group_to_indices.items():
            self.dataset_sizes[ds] = self.dataset_sizes.get(ds, 0) + len(indices)

    def _group_weight(self, group_key, group_len: int) -> float:
        weight = group_len * float(self.role_multipliers.get(group_key[2], 1.0))
        if not np.isfinite(weight):
            raise ValueError("Nonfinite role sampling weight")
        return weight

    def _compute_weights(self) -> Dict[str, float]:
        weights = {}
        role_totals = {}
        for key, indices in self.group_to_indices.items():
            role_totals[key[0]] = role_totals.get(key[0], 0.0) + self._group_weight(
                key, len(indices)
            )
        if not all(np.isfinite(w) for w in role_totals.values()):
            raise ValueError("Nonfinite role sampling weight total")
        active_datasets = {ds for ds, weight in role_totals.items() if weight > 0}
        for ds, size in self.dataset_sizes.items():
            multiplier = float(self.multipliers.get(ds, 1.0))
            if ds not in active_datasets or multiplier == 0:
                weights[ds] = 0.0
                continue
            try:
                weights[ds] = size ** (1.0 / self.temperature) * multiplier
            except OverflowError as exc:
                raise ValueError("Nonfinite dataset sampling weight") from exc
        if not np.isfinite(sum(weights.values())):
            raise ValueError("Nonfinite dataset sampling weight")
        if not any(w > 0 for w in weights.values()):
            raise ValueError("No positive weighted groups")
        return weights

    def _per_dataset_budget(self) -> Dict[str, int]:
        wsum = sum(self.weights.values()) or 1.0
        budget = {}
        for ds in self.dataset_sizes:
            budget[ds] = (
                max(1, int(round(self.total_samples * self.weights[ds] / wsum)))
                if self.weights[ds] > 0
                else 0
            )
        return budget

    def _count_batches(self) -> int:
        # Deterministic batch count (independent of the per-epoch shuffle).
        total = 0
        for ds, n_d in self._per_dataset_n.items():
            res_groups = [
                (k, v) for k, v in self.group_to_indices.items() if k[0] == ds
            ]
            tot_d = sum(self._group_weight(k, len(v)) for k, v in res_groups) or 1.0
            for k, idxs in res_groups:
                weight = self._group_weight(k, len(idxs))
                n_di = (
                    max(1, int(round(n_d * weight / tot_d)))
                    if n_d > 0 and weight > 0
                    else 0
                )
                total += (
                    n_di // self.batch_size
                    if self.drop_last
                    else -(-n_di // self.batch_size)
                )
        return total // self.num_replicas

    def _build_batches(self) -> List[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batches: List[List[int]] = []

        for ds, n_d in self._per_dataset_n.items():
            res_groups = [
                (k, v) for k, v in self.group_to_indices.items() if k[0] == ds
            ]
            tot_d = sum(self._group_weight(k, len(v)) for k, v in res_groups) or 1.0
            for k, idxs in res_groups:
                weight = self._group_weight(k, len(idxs))
                n_di = (
                    max(1, int(round(n_d * weight / tot_d)))
                    if n_d > 0 and weight > 0
                    else 0
                )
                if n_di == 0:
                    continue
                if n_di > len(idxs):
                    pos = torch.randint(len(idxs), (n_di,), generator=g).numpy()
                else:
                    pos = torch.randperm(len(idxs), generator=g)[:n_di].numpy()
                drawn = idxs[pos]
                for start in range(0, len(drawn), self.batch_size):
                    batch = drawn[start : start + self.batch_size]
                    if self.drop_last and len(batch) < self.batch_size:
                        continue
                    batches.append(batch)

        if self.shuffle:
            batches = [
                batches[i] for i in torch.randperm(len(batches), generator=g).tolist()
            ]
        return batches

    def __iter__(self):
        batches = self._build_batches()
        mine = batches[self.rank :: self.num_replicas]
        return (batch.tolist() for batch in mine[: self._num_batches_per_rank])

    def __len__(self) -> int:
        return self._num_batches_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
