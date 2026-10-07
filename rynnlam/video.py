"""Frame-accurate RGB decoding and aspect-preserving input geometry."""

import json
import math
import os
import struct
from pathlib import Path

import cv2
import numpy as np

BUCKETS = {"square": (280, 280), "4x3": (238, 322), "16x9": (210, 364)}

_ROVIDX_PREFIX = "rovidx_tar://"


def pick_bucket(height, width):
    if height <= 0 or width <= 0:
        raise ValueError("Frame dimensions must be positive")
    aspect = math.log(width / height)
    return min(
        BUCKETS,
        key=lambda name: abs(aspect - math.log(BUCKETS[name][1] / BUCKETS[name][0])),
    )


def resize_crop(frame, target_hw):
    height, width = frame.shape[:2]
    th, tw = target_hw
    scale = max(th / height, tw / width)
    nh, nw = int(height * scale + 0.5), int(width * scale + 0.5)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top, left = (nh - th) // 2, (nw - tw) // 2
    return resized[top : top + th, left : left + tw]


def _decode_encoded_frame(frame):
    # Unwrap nested 0-d object arrays (zarr/hdf5 object dtype frames).
    for _ in range(8):
        if isinstance(frame, np.ndarray) and frame.shape == ():
            frame = frame[()]
        else:
            break
    if isinstance(frame, np.ndarray) and frame.dtype != object:
        # Raw RGB frame [height, width, 3].
        if frame.ndim == 3 and frame.shape[-1] == 3:
            return frame
        # JPEG-encoded frame stored as a 1-D uint8 byte buffer (RoboMIND2.0 HDF5).
        if frame.ndim == 1 and frame.dtype == np.uint8:
            image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(frame, (bytes, bytearray, np.bytes_)):
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError("Invalid image frame (expected RGB array or encoded bytes)")


# Backwards-compatible alias for the original zarr-only decoder name.
_decode_zarr_frame = _decode_encoded_frame


class _ZarrV3Array:
    """Minimal reader for zarr v3 variable_length_bytes arrays stored with the
    sharding_indexed(vlen-bytes + zstd) codec (LeRobot v3 exports, e.g. EgoVerse).

    zarr-python 3.x requires Python >= 3.11, which is not available in this
    environment, so shard files are parsed directly: shard 'c/<i>' holds the
    frames, with a trailing index of (offset, length) u64 pairs plus a crc32c.
    """

    def __init__(self, path):
        self._struct = struct
        self.path = Path(path)
        meta = json.loads((self.path / "zarr.json").read_text())
        if meta.get("node_type") != "array":
            raise ValueError(f"Not a zarr v3 array: {path}")
        if meta.get("data_type") != "variable_length_bytes":
            raise ValueError(f"Unsupported zarr v3 data type: {meta.get('data_type')}")
        codecs = meta.get("codecs") or []
        if len(codecs) != 1 or codecs[0].get("name") != "sharding_indexed":
            raise ValueError(f"Unsupported zarr v3 codecs: {codecs}")
        conf = codecs[0]["configuration"]
        inner = [c["name"] for c in conf.get("codecs", [])]
        if inner[:1] != ["vlen-bytes"] or "zstd" not in inner:
            raise ValueError(f"Unsupported inner codecs: {inner}")
        if conf.get("index_location") != "end":
            raise ValueError(f"Unsupported index location: {conf.get('index_location')}")
        self.shape = tuple(meta["shape"])
        self.frames_per_shard = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
        self.sep = meta["chunk_key_encoding"]["configuration"].get("separator", "/")
        self._index_cache = {}

    def _load_index(self, shard_idx):
        cached = self._index_cache.get(shard_idx)
        if cached is not None:
            return cached
        shard_file = self.path / "c" / self.sep.join(str(shard_idx))
        size = shard_file.stat().st_size
        index_size = self.frames_per_shard * 16 + 4
        with open(shard_file, "rb") as handle:
            handle.seek(size - index_size)
            index = handle.read(index_size)
        offsets = [
            self._struct.unpack_from("<Q", index, i * 16)[0]
            for i in range(self.frames_per_shard)
        ]
        lengths = [
            self._struct.unpack_from("<Q", index, i * 16 + 8)[0]
            for i in range(self.frames_per_shard)
        ]
        self._index_cache[shard_idx] = (shard_file, offsets, lengths)
        return self._index_cache[shard_idx]

    def __getitem__(self, index):
        from numcodecs import Zstd

        shard_idx = index // self.frames_per_shard
        inner = index % self.frames_per_shard
        shard_file, offsets, lengths = self._load_index(shard_idx)
        with open(shard_file, "rb") as handle:
            handle.seek(offsets[inner])
            chunk = handle.read(lengths[inner])
        payload = Zstd().decode(chunk)
        # vlen-bytes payload: u32 element count, u32 length, then the bytes.
        count = self._struct.unpack_from("<I", payload, 0)[0]
        if count != 1:
            raise ValueError(f"Expected 1 element per chunk, got {count}")
        length = self._struct.unpack_from("<I", payload, 4)[0]
        return payload[8 : 8 + length]

_rovidx_index_cache = {}


def _rovidx_lookup(url):
    """Return (tar_path, offset, size) for one 'rovidx_tar://<md5>.mp4' clip."""
    tar_root = os.environ.get("ROVIDX_TAR_ROOT")
    index_root = os.environ.get("ROVIDX_INDEX_ROOT")
    if not tar_root or not index_root:
        raise ValueError(
            "rovidx_tar input requires ROVIDX_TAR_ROOT and ROVIDX_INDEX_ROOT"
        )
    name = url[len(_ROVIDX_PREFIX) :]
    shard = name[:2]
    index = _rovidx_index_cache.get(shard)
    if index is None:
        index = {}
        with open(os.path.join(index_root, shard + ".jsonl")) as handle:
            for line in handle:
                entry = json.loads(line)
                # keep-first: some jsonl carry a bogus trailing duplicate whose
                # offset points past the end of the tar.
                index.setdefault(entry["n"], (entry["o"], entry["s"]))
        _rovidx_index_cache[shard] = index
    if name not in index:
        raise FileNotFoundError(f"{name} not found in RoVid-X index {shard}.jsonl")
    offset, size = index[name]
    return os.path.join(tar_root, shard + ".tar"), offset, size


def _rovidx_read_bytes(url):
    tar_path, offset, size = _rovidx_lookup(url)
    with open(tar_path, "rb") as handle:
        handle.seek(offset)
        buffer = handle.read(size)
    if len(buffer) != size:
        raise RuntimeError(f"Short tar read {len(buffer)}/{size} for {url}")
    return buffer


def resolve_source(video_path, base=None):
    """Return (source_id, size_bytes, mtime_ns) for one view's video path.

    Plain files are stat'ed directly; a RoVid-X tar member has no inode of its
    own, so it reports its clip byte size and the shard's mtime instead.
    """
    text = str(video_path)
    if text.startswith(_ROVIDX_PREFIX):
        tar_path, _, size = _rovidx_lookup(text)
        return text, size, os.stat(tar_path).st_mtime_ns
    path = Path(text)
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    path = path.resolve(strict=True)
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


class FrameReader:
    def __init__(self, path, view="head"):
        self.source = str(path)
        self.path = Path(path)
        self.view = view
        self.fps = None
        self.num_frames = None
        if self.source.startswith(_ROVIDX_PREFIX):
            self.backend = "rovidx_tar"
        elif self.path.is_dir():
            self.backend = "zarr"
        elif self.path.suffix.lower() in (".hdf5", ".h5"):
            self.backend = "hdf5"
        else:
            self.backend = "av"

    def frames(self, start=0, end=None):
        if start < 0 or (end is not None and end <= start):
            raise ValueError("Expected a nonempty frame range with start >= 0")
        yielded = 0
        if self.backend == "rovidx_tar":
            import io

            import av

            with av.open(io.BytesIO(_rovidx_read_bytes(self.source))) as container:
                stream = container.streams.video[0]
                self.fps = float(stream.average_rate) if stream.average_rate else None
                for index, frame in enumerate(container.decode(video=0)):
                    if index < start:
                        continue
                    if end is not None and index >= end:
                        break
                    yielded += 1
                    yield frame.to_ndarray(format="rgb24")
        elif self.backend == "zarr":
            import zarr

            array = self._open_zarr_array(zarr)
            total = array.shape[0]
            for index in range(
                start, min(end, total) if end is not None else total
            ):
                yielded += 1
                yield _decode_encoded_frame(array[index])
        elif self.backend == "hdf5":
            import h5py

            # OSS/FUSE does not support POSIX file locking -> locking=False.
            with h5py.File(str(self.path), "r", locking=False) as handle:
                dataset = self._open_hdf5_dataset(handle)
                total = len(dataset)
                stop = min(end, total) if end is not None else total
                for index in range(start, stop):
                    yielded += 1
                    yield _decode_encoded_frame(dataset[index])
        else:
            import av

            with av.open(str(self.path)) as container:
                stream = container.streams.video[0]
                self.fps = float(stream.average_rate) if stream.average_rate else None
                for index, frame in enumerate(container.decode(video=0)):
                    if index < start:
                        continue
                    if end is not None and index >= end:
                        break
                    yielded += 1
                    yield frame.to_ndarray(format="rgb24")
        if yielded == 0 or (end is not None and yielded != end - start):
            raise ValueError(
                f"Decoded {yielded} frames, incomplete interval [{start}, {end}) from {self.path}"
            )
        self.num_frames = yielded

    def _open_zarr_array(self, zarr):
        """Return the zarr image array for this view.

        Handles the EgoVerse layout (a root group holding images.<cam> sub-arrays)
        and the legacy layout (a standalone array sub-directory with its own
        .zarray). When the requested view has no matching array, falls back to the
        first images.* array in the group.
        """
        try:
            root = zarr.open(str(self.path), mode="r")
            # zarr 2.x exposes hierarchy.Group and array_keys(); zarr 3.x uses
            # zarr.Group and keys().
            group_cls = getattr(zarr, "Group", None) or getattr(
                getattr(zarr, "hierarchy", None), "Group", ()
            )

            def _is_array(obj):
                return hasattr(obj, "shape") and not isinstance(obj, group_cls)

            for key in (f"images.{self.view}", self.view):
                try:
                    obj = root[key]
                except Exception:
                    continue
                if _is_array(obj):
                    return obj
            try:
                keys = root.array_keys() if hasattr(root, "array_keys") else root.keys()
                image_keys = sorted(k for k in keys if k.startswith("images."))
            except Exception:
                image_keys = []
            if image_keys:
                return root[image_keys[0]]
            # Legacy layout: a standalone array sub-directory.
            for sub in (f"images.{self.view}", self.view, "images"):
                candidate = self.path / sub
                if (candidate / ".zarray").exists() or (candidate / "zarr.json").exists():
                    return zarr.open(str(candidate), mode="r")
        except Exception:
            pass
        # Fallback for zarr v3 stores the installed zarr-python cannot open.
        for sub in (f"images.{self.view}", self.view, "images"):
            candidate = self.path / sub
            if (candidate / "zarr.json").exists():
                return _ZarrV3Array(candidate)
        for candidate in sorted(self.path.glob("images.*")):
            if (candidate / "zarr.json").exists():
                return _ZarrV3Array(candidate)
        raise ValueError(f"No Zarr image array for view {self.view} in {self.path}")

    def _pick_hdf5_camera(self, cameras):
        """Map the requested view name to an HDF5 color-image camera key."""
        view = self.view.lower()
        aliases = {
            "head": "camera_front",
            "front": "camera_front",
            "global": "camera_front",
            "wrist_left": "camera_left",
            "left": "camera_left",
            "wrist_right": "camera_right",
            "right": "camera_right",
        }
        target = aliases.get(view)
        if target in cameras:
            return target
        for camera in cameras:
            lowered = camera.lower()
            if view == lowered or view in lowered or lowered.endswith(view):
                return camera
        return cameras[0]

    def _open_hdf5_dataset(self, handle):
        """Return the RGB color-image dataset for this view (RoboMIND2.0 layout)."""
        group = "camera_observations/color_images"
        if group not in handle:
            raise ValueError(f"No {group} in {self.path}")
        cameras = list(handle[group].keys())
        if not cameras:
            raise ValueError(f"Empty {group} in {self.path}")
        camera = self._pick_hdf5_camera(cameras)
        return handle[f"{group}/{camera}"]


def pair_batches(
    reader, gap=5, batch_size=16, bucket="auto", start=0, end=None, pair_stride=1
):
    if gap < 1 or batch_size < 1 or pair_stride < 1:
        raise ValueError("gap, batch_size and pair_stride must be positive")
    if bucket != "auto" and bucket not in BUCKETS:
        raise ValueError(f"Unknown bucket {bucket}")
    sources, pairs, indices = {}, [], []
    native_hw = target_hw = None
    chosen = bucket
    count = 0
    for index, raw in enumerate(reader.frames(start, end)):
        if native_hw is None:
            native_hw = raw.shape[:2]
            chosen = pick_bucket(*native_hw) if bucket == "auto" else bucket
            target_hw = BUCKETS[chosen]
        elif raw.shape[:2] != native_hw:
            raise ValueError("Frame resolution changed within the video")
        count += 1
        is_source = index % pair_stride == 0
        is_target = index >= gap and (index - gap) % pair_stride == 0
        if is_source:
            sources[index] = (raw, None)
        if not is_target:
            continue
        source_index = index - gap
        source_raw, source_frame = sources.pop(source_index)
        if source_frame is None:
            source_frame = resize_crop(source_raw, target_hw).astype(np.float32) / 255.0
        frame = resize_crop(raw, target_hw).astype(np.float32) / 255.0
        if is_source:
            sources[index] = (None, frame)
        pairs.append(np.stack([source_frame, frame]))
        indices.append((source_index, index))
        if len(pairs) == batch_size:
            yield np.stack(pairs), np.asarray(indices, dtype=np.int32), chosen, native_hw
            pairs, indices = [], []
    if count <= gap:
        raise ValueError(f"Video interval has {count} frames, but gap={gap}")
    if pairs:
        yield np.stack(pairs), np.asarray(indices, dtype=np.int32), chosen, native_hw