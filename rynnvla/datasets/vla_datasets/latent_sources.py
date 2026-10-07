"""Storage adapters for the RGB sources used by offline latent labeling."""

import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=16)
def _tar_index(index_root, shard):
    entries = {}
    with open(Path(index_root) / f"{shard}.jsonl") as handle:
        for line in handle:
            entry = json.loads(line)
            # Match the labeler: some indices end with invalid duplicate entries.
            entries.setdefault(entry["n"], (int(entry["o"]), int(entry["s"])))
    return entries


class TarVideo(io.RawIOBase):
    """A seekable view of one indexed tar member, without extracting the shard."""

    def __init__(self, uri):
        super().__init__()
        tar_root, index_root = os.getenv("ROVIDX_TAR_ROOT"), os.getenv("ROVIDX_INDEX_ROOT")
        if not tar_root or not index_root:
            raise ValueError("rovidx_tar requires ROVIDX_TAR_ROOT and ROVIDX_INDEX_ROOT")
        name = uri.removeprefix("rovidx_tar://")
        if len(name) < 3 or Path(name).name != name or any(c not in "0123456789abcdef" for c in name[:2]):
            raise ValueError(f"Invalid RoVid-X member: {name!r}")
        shard = name[:2]
        try:
            self.offset, self.size = _tar_index(index_root, shard)[name]
        except KeyError as exc:
            raise FileNotFoundError(f"{name} missing from {index_root}/{shard}.jsonl") from exc
        self.handle = open(Path(tar_root) / f"{shard}.tar", "rb")
        self.position = 0
        if self.offset < 0 or self.size <= 0 or self.offset + self.size > os.fstat(self.handle.fileno()).st_size:
            self.close()
            raise ValueError(f"RoVid-X member range exceeds its tar shard: {uri}")
        self.handle.seek(self.offset)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        origins = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.size}
        if whence not in origins:
            raise ValueError(f"Invalid whence: {whence}")
        position = origins[whence] + offset
        if position < 0:
            raise ValueError("Cannot seek before a tar member")
        self.handle.seek(self.offset + position)
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        size = remaining if size is None or size < 0 else min(size, remaining)
        result = self.handle.read(size)
        if len(result) != size:
            raise OSError(f"Short tar member read: {len(result)}/{size}")
        self.position += len(result)
        return result

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if hasattr(self, "handle"):
            self.handle.close()
        super().close()


def hdf5_camera(cameras, view):
    """Reproduce RynnLAM FrameReader's camera selection for existing latents.

    In particular, ``global`` aliases camera_front and unmatched ``side`` falls
    back to the first stored camera. Changing those choices here would pair the
    already generated latent with a different image. Preserve source view names.
    """
    if not cameras or not view:
        raise ValueError("HDF5 decoding requires a source view and nonempty camera group")
    view = view.lower()
    aliases = {"head": "camera_front", "front": "camera_front", "global": "camera_front",
               "wrist_left": "camera_left", "left": "camera_left",
               "wrist_right": "camera_right", "right": "camera_right"}
    target = aliases.get(view)
    if target in cameras:
        return target
    for camera in cameras:
        lowered = camera.lower()
        if view == lowered or view in lowered or lowered.endswith(view):
            return camera
    return cameras[0]


def read_hdf5_frame(path, index, view):
    import h5py

    with h5py.File(path, "r", locking=False) as handle:
        group = handle["camera_observations/color_images"]
        frames = group[hdf5_camera(list(group.keys()), view)]
        if not 0 <= index < len(frames):
            raise IndexError(f"Frame {index} outside HDF5 stream of length {len(frames)}: {path}")
        frame = frames[index]
    for _ in range(8):
        if isinstance(frame, np.ndarray) and frame.shape == ():
            frame = frame[()]
        else:
            break
    if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[-1] == 3:
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB, got {frame.dtype}: {path}")
        return frame
    if isinstance(frame, np.ndarray) and frame.ndim == 1 and frame.dtype == np.uint8:
        frame = frame.tobytes()
    if not isinstance(frame, (bytes, bytearray, np.bytes_)):
        raise ValueError(f"Expected RGB pixels or encoded image bytes: {path}")
    with Image.open(io.BytesIO(frame)) as image:
        return np.array(image.convert("RGB"))


@lru_cache(maxsize=32)
def _zarr_image_array(path, view):
    """Return (array, key) for one view of a Zarr v3 episode store.

    Reproduces RynnLAM ``FrameReader._open_zarr_array``'s resolution order, because the
    latents were produced through it: ``images.<view>`` then ``<view>``, then the first
    ``images.*`` array in sorted order, then a legacy standalone array sub-directory.
    EgoVerse stores its single camera as ``images.front_1`` while the manifest view is
    named ``head``, so the sorted-``images.*`` fallback is the branch that actually fires;
    reordering these would silently pair a latent with a different camera.

    Cached because a sample decodes several frames of the same episode and opening the
    group costs ~80 ms on the OSS mount -- more than the frame read itself.
    """
    import zarr

    root = zarr.open(path, mode="r")
    group_cls = getattr(zarr, "Group", ())

    def is_array(obj):
        return hasattr(obj, "shape") and not isinstance(obj, group_cls)

    for key in (f"images.{view}", view) if view else ():
        try:
            obj = root[key]
        except Exception:
            continue
        if is_array(obj):
            return obj, key
    try:
        keys = root.array_keys() if hasattr(root, "array_keys") else root.keys()
        image_keys = sorted(k for k in keys if k.startswith("images."))
    except Exception:
        image_keys = []
    if image_keys:
        return root[image_keys[0]], image_keys[0]
    for sub in (f"images.{view}" if view else None, view, "images"):
        if not sub:
            continue
        candidate = Path(path) / sub
        if (candidate / "zarr.json").exists() or (candidate / ".zarray").exists():
            return zarr.open(str(candidate), mode="r"), sub
    raise ValueError(f"No Zarr image array for view {view!r} in {path}")


def read_zarr_frame(path, index, view=None):
    """Decode one frame from a Zarr v3 store, by stream-relative index.

    Frame numbering follows RynnLAM ``FrameReader.frames``, which is what produced the
    latents: it iterates ``range(array.shape[0])`` with **no** cap. Note that the
    alternative RynnLAM decode path clamps to the ``total_frames`` attribute
    instead (2808 for a 2900-slot EgoVerse episode), so the two RynnLAM paths disagree;
    the labeling path is the one the latents' ``num_frames`` records, and all 2900 slots
    decode, so ``shape[0]`` is correct here. Capping at ``total_frames`` would shift every
    frame after it out of alignment with its latent.
    """
    array, _ = _zarr_image_array(path, view)
    total = int(array.shape[0])
    if not 0 <= index < total:
        raise IndexError(f"Frame {index} outside Zarr stream of length {total}: {path}")
    frame = array[index]
    for _ in range(8):
        if isinstance(frame, np.ndarray) and frame.shape == ():
            frame = frame[()]
        else:
            break
    if isinstance(frame, np.ndarray) and frame.dtype != object:
        if frame.ndim == 3 and frame.shape[-1] == 3:
            if frame.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB, got {frame.dtype}: {path}")
            return frame
        if frame.ndim == 1 and frame.dtype == np.uint8:
            frame = frame.tobytes()
    if not isinstance(frame, (bytes, bytearray, np.bytes_)):
        raise ValueError(f"Expected RGB pixels or encoded image bytes: {path}")
    # PIL, matching read_hdf5_frame. RynnLAM decoded with cv2.imdecode, so pixels may
    # differ by +-1 LSB from what the labeler saw -- the same divergence this repo already
    # accepted for HDF5, and far below augmentation noise.
    with Image.open(io.BytesIO(bytes(frame))) as image:
        return np.array(image.convert("RGB"))


# The cache holds live Zarr arrays. A DataLoader worker forks after the parent may have
# decoded a frame, and an inherited store handle is exactly the kind of thing that fails
# rarely and far from its cause -- so the child starts cold and reopens what it needs.
# Measured on an EgoVerse episode: 57.0 ms/frame reopening vs 6.1 ms/frame reusing the
# array, which is why the cache exists at all.
try:
    os.register_at_fork(after_in_child=_zarr_image_array.cache_clear)
except AttributeError:
    pass
