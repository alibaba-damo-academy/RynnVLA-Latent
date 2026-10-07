"""VLABench LeRobot-v3.0 dataset adapter (direct parquet/MP4 reads, no lerobot).

Reads the official VLABench demonstration release (a single LeRobot ``v3.0``
dataset) directly with pyarrow + PyAV.  ``VLABench/meta/info.json`` reports
``robot_type=franka``, ``fps=10``, one ``observation.state[7]`` and one
``action[7]`` feature, and three 224x224 AV1 (libsvtav1) video streams.  Unlike
the LeRobot-v2.1 datasets used elsewhere in this framework, v3.0 shards frames
of many episodes into shared parquet files and concatenates the episode videos
of many episodes into shared MP4 files; the adapter resolves both through the
per-episode pointers in ``meta/episodes/*.parquet``.  Verified against the
actual release: video ``from_timestamp``/``to_timestamp`` are perfectly
contiguous within each shared MP4, so ``round(from_timestamp * fps)`` is the
episode's frame offset in that file.

Action / state semantics (from VLABench ``convert_to_lerobot.py``,
``skill_lib.py`` and ``franka.py``; channel values verified binary {0, 1}):

* state[7]  = ee_pos(3, robot-base-translated frame) + ee_euler_xyz(3) +
  gripper flag(1).  The raw flag comes from ``Franka.get_ee_open_state`` which
  returns 1 when the finger qpos is *below* 0.035 m (i.e. closed) -- a known
  upstream bug.  The adapter inverts it so that 1 = open, matching the action
  channel and the LIBERO convention.
* action[7] = target ee_pos(3) + target ee_euler_xyz(3) + gripper command(1),
  where 1 = open and 0 = closed (finger command binarized at 0.03 m).  Actions
  are absolute robot-frame EE pose targets; the Stage-2 recipe trains them raw
  (``use_delta_action=false``) because naive euler subtraction wraps at +/-pi
  (the euler stats touch +-3.1415).  Rotation is converted to rot_6d and
  normalized with the recipe's ``action_norm_type`` (``min_max_sym`` for
  Stage-2), exactly like ``LiberoPlusDataset``.

Task selection follows the standard protocol: the In-distribution training
set is the ``primitive:`` half (episodes 0-4999 = 10 task families x 500
demonstrations); ``task_category`` selects ``primitive`` (default),
``composite`` or ``all``.  The ``primitive:`` / ``composite:`` prefixes are
stripped from the instruction text.
"""

import json
import logging
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ...constants import (
    CACHE_DIR,
    RobotType,
    RotationRepresentation,
    VIEW_ROLE_TO_ID,
)
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset
from .utils import decode_video_frames

_IMAGE_PREFIX = "observation.images."
_DEFAULT_CAMERAS = ("image", "second_image", "wrist_image")
_TASK_CATEGORY_PREFIXES = {
    "primitive": "primitive:",
    "composite": "composite:",
}
_PREFIX_RE = re.compile(r"^(primitive|composite):\s*")


def _strip_task_prefix(task: str) -> str:
    return _PREFIX_RE.sub("", task.strip()).strip()


def _tasks_to_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        value = value.ravel().tolist()
    return [str(task) for task in value]


def _feature_dim(feature: Dict) -> Optional[int]:
    shape = feature.get("shape")
    if isinstance(shape, list) and len(shape) == 1:
        return int(shape[0])
    return None


def _build_vlabench_action(value: torch.Tensor) -> RobotAction:
    """Convert a float tensor with shape ``(T, 7)`` to canonical fields."""
    if value.ndim != 2 or value.size(1) != 7:
        raise ValueError(f"VLABench action must have shape (T, 7), got {tuple(value.shape)}")
    arm = Arm(
        eef_position=Position(value[:, 0:3]),
        eef_rotation=Rotation(
            value[:, 3:6],
            representation=RotationRepresentation.EULER_XYZ,
        ),
    )
    return RobotAction(
        left_arm=arm,
        # 1 = open, 0 = closed; excluded from delta composition like LIBERO.
        left_gripper=Position(value[:, 6:7], allow_relative=False),
    )


def _build_vlabench_state(value: torch.Tensor) -> RobotState:
    """Convert a float tensor with shape ``(T, 7)`` to canonical fields."""
    if value.ndim != 2 or value.size(1) != 7:
        raise ValueError(f"VLABench state must have shape (T, 7), got {tuple(value.shape)}")
    arm = Arm(
        eef_position=Position(value[:, 0:3]),
        eef_rotation=Rotation(
            value[:, 3:6],
            representation=RotationRepresentation.EULER_XYZ,
        ),
    )
    # Raw flag is 1 when the gripper is closed (upstream get_ee_open_state bug);
    # invert to the framework convention 1 = open so state and action agree.
    gripper = 1.0 - value[:, 6:7]
    return RobotState(
        left_arm=arm,
        left_gripper=Position(gripper, allow_relative=False),
    )


class _PyAVContainerCache:
    """Per-PID LRU cache of open PyAV containers.

    VLABench v3.0 concatenates thousands of episodes into ~500 MB MP4s on a
    FUSE mount: reopening the container per read costs ~0.1 s, while a seek +
    decode on a warm container costs ~3 ms.  Keeping containers open (like
    LiberoPlusDataset's HDF5 handle cache) is what makes random access viable.
    """

    def __init__(self, max_size: int = 32):
        self.max_size = max_size
        self._caches: Dict[int, OrderedDict] = {}

    def get(self, video_path: str):
        import av

        pid = os.getpid()
        cache = self._caches.setdefault(pid, OrderedDict())
        if video_path in cache:
            cache.move_to_end(video_path)
            return cache[video_path]

        container = av.open(video_path)
        cache[video_path] = container
        if len(cache) > self.max_size:
            _, evicted = cache.popitem(last=False)
            evicted.close()
        return container

    def clear(self) -> None:
        for cache in self._caches.values():
            for container in cache.values():
                try:
                    container.close()
                except Exception:
                    pass
        self._caches.clear()


@DATASET_REGISTRY.register()
class VLABenchDataset(BaseVLADataset):
    """The official VLABench demonstrations (LeRobot v3.0, franka, 10 fps).

    Additional arguments:
        task_category: ``primitive`` (default; the 10-task In-distribution
            training set, 500 demos per task), ``composite`` or ``all``.
        cameras: optional camera-name whitelist.  Names may include or omit
            the ``observation.images.`` prefix; default selects image,
            second_image, wrist_image.
        video_backend: ``pyav`` (default; persistent container cache, handles
            the AV1 streams) or ``torchcodec``.  Beware: torchcodec builds a
            frame index per ~500 MB file and costs ~20 s per cold open on the
            FUSE mount.
        container_cache_capacity: open-video-container LRU size (pyav only).
        video_cache_dir: local directory that receives a one-time copy of the
            referenced video files (default: ``<CACHE_DIR>/vlabench_videos``).
            The FUSE mount's daemon-side cache is smaller than even the
            primitive subset, so random reads thrash back to ~0.7 s per seek;
            local copies make every read a millisecond-scale random access.
            Files whose copy fails keep reading from the mount.
        prewarm_videos: when the local cache is unusable, stream-read every
            referenced video once at init to warm the mount's daemon cache
            (best effort; helps only while the subset stays cached).
        prewarm_workers: parallel readers used by the cache/prewarm pass.
        strict: validate schema and fail on missing videos.  Default True.
    """

    # Super's VIEW_ROLES split the generic side camera into side_left(4) /
    # side_right(5); constants.SERVER_CAMERA_SLOT_MAP routes a single "side"
    # camera to the primary (left) slot, so VLABench's second_image -> side_left.
    _CAMERA_SLOTS = {
        "image": VIEW_ROLE_TO_ID["front_third"],
        "second_image": VIEW_ROLE_TO_ID["side_left"],
        "wrist_image": VIEW_ROLE_TO_ID["left_wrist"],
    }

    def __init__(
        self,
        data_path: str,
        action_chunk_size: int,
        use_delta_action: bool,
        task_category: str = "primitive",
        cameras: Optional[List[str]] = None,
        video_backend: str = "pyav",
        container_cache_capacity: int = 32,
        prewarm_videos: bool = True,
        prewarm_workers: int = 8,
        video_cache_dir: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        if task_category not in ("primitive", "composite", "all"):
            raise ValueError("task_category must be 'primitive', 'composite' or 'all'")
        if video_backend not in ("pyav", "torchcodec"):
            raise ValueError("video_backend must be 'pyav' or 'torchcodec'")
        if container_cache_capacity <= 0:
            raise ValueError("container_cache_capacity must be positive")

        self.task_category = task_category
        self.video_backend = video_backend
        self.strict = bool(strict)
        self._camera_filter = tuple(
            name[len(_IMAGE_PREFIX):] if name.startswith(_IMAGE_PREFIX) else name
            for name in cameras
        ) if cameras else None

        root = Path(os.path.abspath(os.path.expanduser(data_path)))
        info = self._load_info(root)
        self._validate_info(info, root / "meta" / "info.json")

        available_cameras = sorted(
            key[len(_IMAGE_PREFIX):]
            for key, feature in info["features"].items()
            if key.startswith(_IMAGE_PREFIX) and feature.get("dtype") == "video"
        )
        if self._camera_filter is None:
            cameras_sel = [name for name in _DEFAULT_CAMERAS if name in available_cameras]
            cameras_sel.extend(name for name in available_cameras if name not in cameras_sel)
        else:
            missing = sorted(set(self._camera_filter) - set(available_cameras))
            if missing and self.strict:
                raise ValueError(f"Cameras missing from {root / 'meta' / 'info.json'}: {missing}")
            cameras_sel = [name for name in self._camera_filter if name in available_cameras]
        if not cameras_sel:
            raise ValueError(f"No selected video cameras in {root / 'meta' / 'info.json'}")

        # Must precede _load_episodes, which resolves one video pointer per camera.
        self.cameras = tuple(sorted(cameras_sel))
        self._root = root
        self._fps = float(info["fps"])

        self._episodes = self._load_episodes(root, info, task_category)
        if not self._episodes:
            raise ValueError(f"No episodes with task_category={task_category!r} in {root}")

        # BaseVLADataset uses the presence of camera_slot_map to activate its
        # multi-view path.  Current processor ordering is sorted camera name.
        self.camera_slot_map = {
            camera: self._CAMERA_SLOTS.get(camera, index)
            for index, camera in enumerate(self.cameras)
        }

        self._load_action_state_tables()

        self._containers = _PyAVContainerCache(max_size=container_cache_capacity)
        cache_dir = (
            Path(os.path.abspath(os.path.expanduser(video_cache_dir)))
            if video_cache_dir is not None
            else Path(CACHE_DIR) / "vlabench_videos"
        )
        if not self._materialize_video_cache(cache_dir, prewarm_workers) and prewarm_videos:
            self._prewarm_videos(prewarm_workers)

        super().__init__(
            data_path=str(root),
            action_chunk_size=action_chunk_size,
            use_delta_action=use_delta_action,
            **kwargs,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Open av containers hold fds and must not cross a fork/spawn boundary.
        state["_containers"] = _PyAVContainerCache(max_size=self._containers.max_size)
        return state

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_info(root: Path) -> Dict:
        import pyarrow.parquet as pq  # noqa: F401  (fail early if missing)

        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"Missing LeRobot v3.0 metadata: {info_path}. VLABenchDataset expects "
                "the dataset root that directly contains meta/info.json."
            )
        with info_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _validate_info(self, info: Dict, info_path: Path) -> None:
        errors = []
        if info.get("codebase_version") != "v3.0":
            errors.append(f"codebase_version={info.get('codebase_version')!r}, expected 'v3.0'")
        if str(info.get("robot_type", "")).lower() != "franka":
            errors.append(f"robot_type={info.get('robot_type')!r}, expected 'franka'")
        features = info.get("features", {})
        if _feature_dim(features.get("observation.state", {})) != 7:
            errors.append("observation.state must have shape [7]")
        if _feature_dim(features.get("action", {})) != 7:
            errors.append("action must have shape [7]")
        if not features or not any(
            key.startswith(_IMAGE_PREFIX) and feature.get("dtype") == "video"
            for key, feature in features.items()
        ):
            errors.append("no observation.images.* video features")
        if errors and self.strict:
            raise ValueError(f"Unsupported VLABench schema in {info_path}: {'; '.join(errors)}")

    def _load_episodes(self, root: Path, info: Dict, task_category: str) -> List[Dict]:
        """Read meta/episodes/*.parquet and keep the selected task category."""
        import pyarrow.parquet as pq

        episodes_root = root / "meta" / "episodes"
        episode_files = sorted(episodes_root.glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode metadata under {episodes_root}")

        video_template = info.get(
            "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        fps = float(info["fps"])
        columns = ["episode_index", "tasks", "length"]
        for camera in self.cameras:
            columns.extend(
                f"videos/{_IMAGE_PREFIX}{camera}/{field}"
                for field in ("chunk_index", "file_index", "from_timestamp")
            )

        prefix = _TASK_CATEGORY_PREFIXES.get(task_category)
        selected: List[Dict] = []
        for path in episode_files:
            table = pq.read_table(path, columns=columns)
            missing = [name for name in columns if name not in table.column_names]
            if missing and self.strict:
                raise ValueError(f"Episode metadata {path} lacks columns {missing}")
            episode_indices = table.column("episode_index").to_pylist()
            tasks_col = table.column("tasks").to_pylist()
            lengths = table.column("length").to_pylist()
            pointer_columns = {
                camera: (
                    table.column(f"videos/{_IMAGE_PREFIX}{camera}/chunk_index").to_pylist(),
                    table.column(f"videos/{_IMAGE_PREFIX}{camera}/file_index").to_pylist(),
                    table.column(f"videos/{_IMAGE_PREFIX}{camera}/from_timestamp").to_pylist(),
                )
                for camera in self.cameras
            }
            for row in range(table.num_rows):
                tasks = [
                    task for task in _tasks_to_list(tasks_col[row]) if task.strip()
                ]
                if not tasks:
                    continue
                if prefix is not None and not tasks[0].startswith(prefix):
                    continue
                length = int(lengths[row])
                if length <= 0:
                    continue
                episode = {
                    "episode_index": int(episode_indices[row]),
                    "length": length,
                    "instruction": _strip_task_prefix(tasks[0]),
                }
                for camera in self.cameras:
                    chunk_index, file_index, from_ts = pointer_columns[camera]
                    video_key = _IMAGE_PREFIX + camera
                    episode[f"{camera}/video_path"] = str(root / video_template.format(
                        chunk_index=int(chunk_index[row]),
                        file_index=int(file_index[row]),
                        video_key=video_key,
                    ))
                    # v3.0 concatenates episodes contiguously into shared MP4s:
                    # from_timestamp * fps is the episode's first frame index.
                    episode[f"{camera}/frame_offset"] = int(round(float(from_ts[row]) * fps))
                selected.append(episode)
        selected.sort(key=lambda episode: episode["episode_index"])
        return selected

    # ------------------------------------------------------------------
    # Local video cache / prewarm
    # ------------------------------------------------------------------
    def _materialize_video_cache(self, cache_dir: Path, workers: int) -> bool:
        """Copy the referenced videos to a local directory; repoint episode paths.

        The dataset lives on a FUSE mount whose daemon-side data cache is
        smaller than even the primitive subset (~4 GB), so random reads
        thrash back to ~0.7 s per seek.  A one-time copy to local disk
        (primitive: ~4 GB, ~1 min) makes every later read a cheap local
        random access.  Returns False when the cache directory is unusable,
        in which case episode paths keep pointing at the mount.
        """
        import shutil
        import time
        from concurrent.futures import ThreadPoolExecutor

        sources = sorted({
            episode[f"{camera}/video_path"]
            for episode in self._episodes
            for camera in self.cameras
        })
        relative = {source: Path(source).relative_to(self._root) for source in sources}

        logger = logging.getLogger(__name__)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Video cache dir %s unusable (%s); reading videos from %s directly",
                cache_dir, exc, self._root,
            )
            return False

        def _copy(source: str) -> int:
            destination = cache_dir / relative[source]
            try:
                source_size = Path(source).stat().st_size
                if destination.is_file() and destination.stat().st_size == source_size:
                    return 0
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(f"{destination.name}.part{os.getpid()}")

                def _do_copy():
                    shutil.copyfile(source, partial)
                    os.replace(partial, destination)

                self._retry(_do_copy)
            except OSError as exc:
                logger.warning(
                    "Copying %s to the local video cache failed (%s); "
                    "reading this file from the mount", source, exc,
                )
                try:
                    partial.unlink()
                except (OSError, UnboundLocalError):
                    pass
                return 0
            return source_size

        start = time.time()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            copied = sum(executor.map(_copy, sources))
        if copied:
            logger.info(
                "Cached %.2f GB of VLABench videos in %s (%.0f MB/s)",
                copied / 1e9, cache_dir, copied / 1e6 / max(time.time() - start, 1e-9),
            )
        else:
            logger.info("VLABench video cache up to date at %s", cache_dir)

        for episode in self._episodes:
            for camera in self.cameras:
                source = episode[f"{camera}/video_path"]
                destination = cache_dir / relative[source]
                if destination.is_file():
                    episode[f"{camera}/video_path"] = str(destination)
        return True

    def _prewarm_videos(self, workers: int) -> None:
        """Stream-read every referenced video once to warm the OS page cache.

        The dataset lives on a FUSE mount: a cold random seek into one of the
        shared ~200 MB MP4s costs ~0.7 s of network round trips, while a seek
        that hits the page cache costs ~3 ms.  The selected episodes reference
        a bounded set of files (primitive: 3 cameras x 8 files ~= 5 GB), so
        one sequential pass at init removes the cold-read penalty for the
        whole run.  The page cache is shared across processes, so dataloader
        workers forked after this point inherit warm reads for free.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        paths = sorted({
            episode[f"{camera}/video_path"]
            for episode in self._episodes
            for camera in self.cameras
        })
        logger = logging.getLogger(__name__)
        logger.info(
            "Prewarming %d VLABench video files (task_category=%s) with %d readers",
            len(paths), self.task_category, workers,
        )

        def _read(path: str) -> int:
            total = 0
            try:
                with open(path, "rb") as stream:
                    while True:
                        chunk = stream.read(8 << 20)
                        if not chunk:
                            break
                        total += len(chunk)
            except OSError as exc:
                logger.warning("Prewarm failed for %s: %s", path, exc)
            return total

        start = time.time()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            bytes_read = sum(executor.map(_read, paths))
        elapsed = max(time.time() - start, 1e-9)
        logger.info(
            "Prewarmed %d VLABench video files (%.2f GB) in %.1f s (%.0f MB/s)",
            len(paths), bytes_read / 1e9, elapsed, bytes_read / 1e6 / elapsed,
        )

    # ------------------------------------------------------------------
    # Frame table loading (action / state)
    # ------------------------------------------------------------------
    def _load_action_state_tables(self) -> None:
        """Materialize per-episode action/state arrays once at init.

        v3.0 shards ~1.5 M frames per parquet file, so a filtered read per
        training sample would re-read a ~65 MB row group every time.  All
        frames of the selected category fit in RAM (primitive: ~575k frames x
        14 floats x 4 B = ~32 MB), and numpy buffers stay copy-on-write shared
        with forked dataloader workers.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        lengths = [episode["length"] for episode in self._episodes]
        total_frames = int(sum(lengths))
        self._action_table = np.zeros((total_frames, 7), dtype=np.float32)
        self._state_table = np.zeros((total_frames, 7), dtype=np.float32)
        self._episode_row_offsets = np.zeros(len(self._episodes) + 1, dtype=np.int64)
        self._episode_row_offsets[1:] = np.cumsum(lengths)

        episode_ids = [episode["episode_index"] for episode in self._episodes]
        assert min(episode_ids) >= 0
        row_of_episode = np.full(max(episode_ids) + 1, -1, dtype=np.int64)
        for slot, episode_id in enumerate(episode_ids):
            row_of_episode[episode_id] = slot
        length_of_slot = np.asarray(lengths, dtype=np.int64)
        offsets_of_slot = self._episode_row_offsets[:-1].copy()
        min_episode, max_episode = min(episode_ids), max(episode_ids)

        def _flat_values(batch, name: str) -> np.ndarray:
            column = batch.column(name)
            if isinstance(column, pa.ChunkedArray):
                column = column.combine_chunks()
            values = np.asarray(
                column.flatten().to_numpy(zero_copy_only=False), dtype=np.float32
            )
            if values.size != batch.num_rows * 7:
                raise ValueError(
                    f"{name} column flattened to {values.size} values for "
                    f"{batch.num_rows} rows; expected {batch.num_rows * 7} "
                    "(nulls or non-7-dim rows in the parquet)"
                )
            return values.reshape(-1, 7)

        # Scan the shared data shards; row groups whose episode_index stats do
        # not overlap the selected id range are skipped without reading.
        filled = np.zeros(total_frames, dtype=bool)
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            parquet_file = pq.ParquetFile(path)
            schema_names = parquet_file.schema.names
            if "episode_index" not in schema_names:
                raise ValueError(f"Missing episode_index column in {path}")
            episode_col = schema_names.index("episode_index")
            row_groups = []
            for rg in range(parquet_file.metadata.num_row_groups):
                stats = parquet_file.metadata.row_group(rg).column(
                    episode_col
                ).statistics
                if stats is None or not stats.has_min_max:
                    row_groups.append(rg)
                    continue
                if stats.max < min_episode or stats.min > max_episode:
                    continue
                row_groups.append(rg)
            if not row_groups:
                continue
            for batch in parquet_file.iter_batches(
                batch_size=65536,
                row_groups=row_groups,
                columns=["action", "observation.state", "episode_index", "frame_index"],
            ):
                epi = batch.column("episode_index").to_numpy()
                in_range = (epi >= min_episode) & (epi <= max_episode)
                slots = np.where(
                    in_range,
                    row_of_episode[np.clip(epi, 0, row_of_episode.shape[0] - 1)],
                    -1,
                )
                valid = in_range & (slots >= 0)
                if not valid.any():
                    continue
                frame_idx = batch.column("frame_index").to_numpy()
                slots_v = slots[valid]
                frame_v = frame_idx[valid]
                if np.any(frame_v >= length_of_slot[slots_v]) or np.any(frame_v < 0):
                    raise ValueError(
                        f"Frame index outside episode length in {path} "
                        f"(episodes {epi[valid][:5]}...)"
                    )
                rows = offsets_of_slot[slots_v] + frame_v
                action_flat = _flat_values(batch, "action")
                state_flat = _flat_values(batch, "observation.state")
                self._action_table[rows] = action_flat[valid]
                self._state_table[rows] = state_flat[valid]
                filled[rows] = True

        if not filled.all():
            missing = int((~filled).sum())
            raise ValueError(
                f"{missing}/{total_frames} frames of the selected episodes were not "
                f"found in {self._root / 'data'}; the parquet shards and "
                "meta/episodes metadata disagree."
            )

    # ------------------------------------------------------------------
    # BaseVLADataset API
    # ------------------------------------------------------------------
    @property
    def episode_lengths(self) -> List[int]:
        return [episode["length"] for episode in self._episodes]

    def get_robot_type(self, episode_index: int) -> RobotType:
        return RobotType.FRANKA

    def get_fps(self, episode_index: int) -> float:
        return self._fps

    def _index_lengths(self):
        # One index per CHUNK START, not per frame -- same override LiberoPlusDataset
        # makes. The base default enumerates every target frame while __getitem__
        # snaps onto chunk_stride, so with the Stage-2 recipe's action_chunk_size=10 +
        # chunk_overlap_ratio=0.5 (stride 5) len() would overcount ~5x: each distinct
        # chunk handed back five times per epoch, inflating num_update_steps_per_epoch
        # and breaking the resume batch-skip math.
        return self._chunk_start_counts()

    def _indices(
        self,
        episode_index: int,
        frame_index: Union[int, List[int], slice],
    ) -> List[int]:
        length = self._episodes[episode_index]["length"]
        if isinstance(frame_index, int):
            indices = [frame_index]
        elif isinstance(frame_index, slice):
            indices = list(range(*frame_index.indices(length)))
        else:
            indices = [int(index) for index in frame_index]
        if any(index < 0 or index >= length for index in indices):
            raise IndexError(f"Frame index outside episode length {length}: {indices}")
        return indices

    def _episode_rows(self, episode_index: int, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        offset = self._episode_row_offsets[episode_index]
        rows = offset + np.asarray(indices, dtype=np.int64)
        return self._action_table[rows], self._state_table[rows]

    def load_action(
        self,
        episode_index: int,
        frame_index: Union[int, List[int], slice],
    ) -> RobotAction:
        indices = self._indices(episode_index, frame_index)
        actions, _ = self._episode_rows(episode_index, indices)
        return _build_vlabench_action(torch.from_numpy(actions.copy()))

    def load_state(
        self,
        episode_index: int,
        frame_index: Union[int, List[int], slice],
    ) -> RobotState:
        indices = self._indices(episode_index, frame_index)
        _, states = self._episode_rows(episode_index, indices)
        return _build_vlabench_state(torch.from_numpy(states.copy()))

    def load_images(
        self,
        episode_index: int,
        frame_index: Union[int, List[int], slice],
    ) -> Dict[str, torch.Tensor]:
        indices = self._indices(episode_index, frame_index)
        episode = self._episodes[episode_index]
        length = episode["length"]

        # Some episodes contain undecodable frames: on a total decode failure,
        # retry the same indices. Never change images without reloading action/state.
        max_attempts = 10
        for attempt in range(max_attempts):
            images: Dict[str, torch.Tensor] = {}
            for camera in self.cameras:
                video_path = episode[f"{camera}/video_path"]
                if not Path(video_path).is_file():
                    if self.strict:
                        raise FileNotFoundError(f"Missing VLABench video: {video_path}")
                    continue
                video_indices = [episode[f"{camera}/frame_offset"] + index for index in indices]
                try:
                    if self.video_backend == "pyav":
                        frames = self._retry(
                            lambda: self._decode_frames_pyav(video_path, video_indices)
                        )
                    else:
                        frames = self._retry(
                            lambda: decode_video_frames(
                                video_path, video_indices, backend=self.video_backend
                            )
                        )
                        # Contiguous float conversion first; the permuted view
                        # is free (a strided .float() costs ~2.5 ms per frame
                        # on many-core hosts).
                        frames = frames.float().div_(255.0).permute(0, 3, 1, 2)
                except Exception:
                    if self.strict:
                        images.clear()
                        break
                    continue
                images[camera] = frames
            if images or not self.strict:
                return images
            if attempt < max_attempts - 1:
                logging.getLogger(__name__).warning(
                    "Camera decode failed for episode %d (attempt %d/%d); retrying the same frame indices %s",
                    episode_index, attempt + 1, max_attempts, indices,
                )
        raise RuntimeError(f"No camera frame decoded for episode {episode_index}")

    def _decode_frames_pyav(self, video_path: str, indices: List[int]) -> torch.Tensor:
        """Seek + decode on a cached container.  Returns ``(T, C, H, W)`` [0,1]."""
        container = self._containers.get(video_path)
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        time_base = float(stream.time_base)

        frames: List[Optional[np.ndarray]] = []
        last_good: Optional[np.ndarray] = None
        for index in indices:
            pts = int(round(index / rate / time_base))
            container.seek(pts, stream=stream, any_frame=False, backward=True)
            decoded = None
            fallback = None
            for frame in container.decode(stream):
                arr = frame.to_ndarray(format="rgb24")
                fallback = arr
                position = (
                    int(round(frame.pts * time_base * rate))
                    if frame.pts is not None
                    else index
                )
                if position >= index:
                    decoded = arr
                    break
            # Indices past the end of the video clamp to the last frame seen.
            if decoded is None:
                decoded = fallback if fallback is not None else last_good
            if decoded is not None:
                last_good = decoded
                frames.append(decoded)
            else:
                frames.append(None)
        if frames[0] is None:
            raise RuntimeError(f"No frames decoded from {video_path}")
        # Convert in numpy: a permuted uint8 -> float tensor op costs ~2.5 ms
        # on many-core hosts (OpenMP sync for a strided 150 KB copy), while the
        # numpy equivalent is ~0.03 ms.  Output stays (T, C, H, W) float [0, 1].
        array = np.stack(frames).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(0, 3, 1, 2)

    @staticmethod
    def _retry(fn, max_retries: int = 3, delay: int = 2):
        """Retry transient video/parquet I/O like ``RoboCasaDataset``."""
        import time

        for attempt in range(max_retries):
            try:
                return fn()
            except (OSError, IOError):
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise

    def load_instruction(
        self,
        episode_index: int,
        frame_index: Union[int, List[int], slice],
    ) -> List[str]:
        count = len(self._indices(episode_index, frame_index))
        return [self._episodes[episode_index]["instruction"]] * count

    def load_episode(self, episode_index: int, action_only: bool) -> Dict:
        length = self._episodes[episode_index]["length"]
        full_episode = slice(0, length)
        output = {
            "action": self.load_action(episode_index, full_episode),
            "state": self.load_state(episode_index, full_episode),
        }
        if not action_only:
            output["text"] = self.load_instruction(episode_index, full_episode)
            output["images"] = self.load_images(episode_index, full_episode)
        return output

    def __getitem__(self, index: int):
        # A broken sample (missing video file, corrupt parquet, truncated
        # network read) is replaced by a fresh random sample instead of killing
        # the dataloader worker.
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                return super().__getitem__(index)
            except Exception:
                if attempt >= max_attempts - 1:
                    raise
                episode_index = self._resolve_index(index)[0]
                new_index = random.randint(0, len(self) - 1)
                logging.getLogger(__name__).warning(
                    "VLABench sample load failed (episode %d, attempt %d/%d), "
                    "retrying with random sample %d",
                    episode_index, attempt + 1, max_attempts, new_index,
                )
                index = new_index

    def _schema_cache_key(self) -> Dict:
        key = super()._schema_cache_key()
        key.update({
            "task_category": self.task_category,
            "cameras": list(self.cameras),
            "schema_version": 1,
        })
        return key
