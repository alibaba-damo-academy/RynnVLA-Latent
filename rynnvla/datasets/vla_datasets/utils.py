import os
from collections import OrderedDict
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════
#  Video frame decoding (LeRobot-style: torchcodec + pyav backends)
# ═══════════════════════════════════════════════════════════════════

class VideoDecoderCache:
    """Per-PID LRU cache for torchcodec ``VideoDecoder`` instances.

    A single video file may be read many times across different timesteps
    during training.  Re-creating the decoder every time is expensive because
    it has to re-parse the container header.  This cache keeps the *N* most
    recently used decoders alive (per worker process) so sequential reads
    from the same episode are essentially free.

    The cache is keyed by ``(video_path, *extra_key)`` so callers that open
    the same physical file with different logical identifiers can still share
    or separate decoders as needed.
    """

    def __init__(self, max_size: int = 64):
        self.max_size = max_size
        self._caches: Dict[int, OrderedDict] = {}
        self._lock = Lock()

    def get(self, video_path: str, *extra_key) -> "VideoDecoder":
        from torchcodec.decoders import VideoDecoder

        pid = os.getpid()
        key = (video_path, *extra_key)

        with self._lock:
            cache = self._caches.setdefault(pid, OrderedDict())
            if key in cache:
                cache.move_to_end(key)
                return cache[key]

        dec = VideoDecoder(video_path)

        with self._lock:
            cache = self._caches.setdefault(pid, OrderedDict())
            cache[key] = dec
            if len(cache) > self.max_size:
                cache.popitem(last=False)

        return dec

    def get_or_none(self, video_path: str, *extra_key) -> Optional["VideoDecoder"]:
        if not os.path.exists(video_path):
            return None
        return self.get(video_path, *extra_key)

    def clear(self):
        with self._lock:
            self._caches.clear()


_default_decoder_cache = VideoDecoderCache()


def decode_video_frames_torchcodec(
    video_path: str,
    indices: List[int],
    cache: Optional[VideoDecoderCache] = None,
) -> torch.Tensor:
    """Decode frames by index using torchcodec.

    Returns
    -------
    torch.Tensor
        ``(T, H, W, C)`` uint8
    """
    if cache is None:
        cache = _default_decoder_cache
    dec = cache.get(video_path)
    frames = dec.get_frames_at(indices=indices).data  # (T, C, H, W) uint8
    return frames.permute(0, 2, 3, 1).contiguous()


def decode_video_frames_pyav(
    video_path: str,
    indices: List[int],
) -> torch.Tensor:
    """Decode frames by index using PyAV (seek + forward-decode).

    Assumes CFR.  *indices* must be monotonically non-decreasing.

    Returns
    -------
    torch.Tensor
        ``(T, H, W, C)`` uint8
    """
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "FRAME"
        rate = float(stream.average_rate)
        time_base = float(stream.time_base)
        # Frame indices are stream-relative, but PTS need not start at zero: EPIC_KITCHENS'
        # split clips carry start_time=50700, which made frame 0 resolve to pos 51 and the
        # decode abort with IndexError. No-op for streams that start at zero.
        start_pts = int(stream.start_time or 0)

        first = int(indices[0])
        last = int(indices[-1])
        pts = start_pts + int(round(first / rate / time_base))
        container.seek(pts, stream=stream, any_frame=False, backward=True)

        wanted = set(int(i) for i in indices)
        out: Dict[int, np.ndarray] = {}
        pos = -1
        for frame in container.decode(stream):
            if pos < 0:
                pos = (
                    int(round((frame.pts - start_pts) * time_base * rate))
                    if frame.pts is not None
                    else 0
                )
            else:
                pos += 1
            if pos < first:
                continue
            if pos in wanted:
                out[pos] = frame.to_ndarray(format="rgb24")
                if len(out) == len(wanted):
                    break
            if pos > last:
                break

        if len(out) != len(wanted):
            missing = sorted(wanted - out.keys())
            raise IndexError(
                f"Could not decode frames {missing} from {video_path}"
            )
        arr = np.stack([out[int(i)] for i in indices], axis=0)
        return torch.from_numpy(arr)
    finally:
        container.close()


def decode_video_frames(
    video_path: str,
    indices: List[int],
    backend: str = "pyav",
    cache: Optional[VideoDecoderCache] = None,
) -> torch.Tensor:
    """Unified video frame decoder (index-based).

    Parameters
    ----------
    video_path : str
        Path to the MP4 file.
    indices : list of int
        Frame indices to extract (0-based).  Must be monotonically
        non-decreasing for the ``"pyav"`` backend.
    backend : ``"pyav"`` | ``"torchcodec"``
        Decoding backend.  ``"pyav"`` is the default because PyAV is a hard
        dependency of this package and every shipped dataset uses it.
        ``"torchcodec"`` is faster for random frame access and keeps an LRU
        decoder cache, but it is an OPTIONAL, undeclared dependency that is
        ABI-paired with torch -- install the release whose compatibility table
        lists your torch version, and do not select this backend otherwise.
    cache : VideoDecoderCache, optional
        Custom cache instance (torchcodec only).  Uses the module-level
        default cache if *None*.

    Returns
    -------
    torch.Tensor
        ``(T, H, W, C)`` uint8
    """
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, indices, cache)
    elif backend == "pyav":
        return decode_video_frames_pyav(video_path, indices)
    else:
        raise ValueError(f"Unknown video backend: {backend!r}")

