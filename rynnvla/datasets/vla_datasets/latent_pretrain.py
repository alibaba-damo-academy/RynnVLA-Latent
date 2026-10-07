"""Latent-action pretraining dataset (Stage-1, no robot actions).

Reads the manifest produced by ``scripts/rebuild_latent_manifest.py``
(``pretrain-latent/vla_train.json``): one entry per episode, each with a caption and
one or more views (video + per-view ``latent.npz``). A sample is

    images   : one frame per view, taken at source frame ``start_frame + start * pair_stride``
    text     : the episode caption
    latents  : ``latent_action[start : start + chunk*stride : stride]`` per view

``start`` indexes the LATENT array, not the video. Latent ``i`` covers source frames
``(start_frame + i*pair_stride, start_frame + i*pair_stride + gap)``, so ``pair_stride`` (per
episode, from the manifest's views; 1 for the frame-aligned LAM extractor, 4 for RynnLAM's
stride-4 labeling) is what converts a latent index back to a frame, and ``start_frame`` is the
labeling window's offset into the source video -- zero for most corpora, nonzero for those
labeled as consecutive windows of one long recording. ``stride`` is likewise a
latent-index step, derived per episode as ``round(fps * latent_step_seconds /
pair_stride)`` so one latent step spans ``latent_step_seconds`` regardless of ``P``.

and *no* action/state — the model's multiview latent branch supervises the action expert
with flow matching against these latent actions alone, so human video works.

Two things are dataset-specific and must not be uniform across the five sources:

* **View ordering.** Views are packed in a canonical order derived from the
  ``constants.VIEW_ROLES`` role of each camera (head first, then wrists, then third-person),
  so "the first image" means the same kind of camera in every source instead of whatever
  sorts first alphabetically: EgoDex ``ego`` and RoboMIND ``camera_top`` are both head views,
  while SthSthV2's ``main`` is third-person. ``_VIEW_ROLE`` below is the (dataset, view name)
  -> role table. The *slot index* IS that role id — the same fixed axis stage-2 uses
  (``LiberoPlusDataset.camera_slot_map``), so a camera means the same slot in both stages.

  This replaced a dense sample-order packing whose stated reason ("feeding role ids as slot
  ids would silently zero the seeds of any view whose role >= K") lapsed when the old ``top``
  role was merged into ``head``: ``VIEW_ROLES`` is a closed enumeration and ``_load_index``
  asserts every cached role indexes it. Dense packing was
  actively harmful for camera identity: ``views`` is a dict so roles never collide, but 94.2%
  of episodes carry a single camera and dense packing sent every one of them to slot 0
  regardless of whether that camera was a head, a wrist or a third-person view — while
  roles 3 (``front_third``, 47,398 views) and 4 (``side``) received no gradient at all.
  Role 3 is exactly what LIBERO's front camera maps to downstream.
* **Sampling rate.** LAM extracted one latent per source frame with a fixed ``gap=5``, but
  the sources run at 12 (SthSthV2), 20 (RoboMIND), 30 (EgoDex/RDT-1B) and 50/59.94 fps
  (EPIC, mixed within the dataset). A uniform latent stride would make one latent step span
  0.083 s on EPIC and 0.42 s on SthSthV2 — 5x apart, against a downstream LIBERO step of
  0.25 s (20 Hz / 5 frames). The stride is therefore derived per episode from that episode's
  own fps so every latent step spans ``latent_step_seconds``; RoboMIND lands back on 5.

The manifest is 281 MB of JSON (1.7 GB of Python objects, ~90 s to parse) which every rank
would otherwise pay, so ``build_index`` flattens it into a numpy/blob ``.npz`` that loads in
seconds; ``scripts/build_latent_pretrain_index.py`` writes it. A full-corpus manifest is
gigabytes of JSON and ``json.load`` would need roughly six times that in RSS, so ``build_index``
streams its input and packs the string columns incrementally -- see ``manifest_io``.
"""
import json
import os
import shutil
import tempfile
import time
from array import array
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import get_worker_info

from ...constants import NUM_VIEW_SLOTS, VIEW_ROLES, VIEW_NAME_TO_ROLE, VIEW_ROLE_TO_ID, RobotType
from ...registry import DATASET_REGISTRY
from ...utils.logging import get_logger
from ...utils.manifest_io import iter_manifest, manifest_format
from .base import (
    BaseVLADataset,
    _apply_visual_augmentation,
    _sample_visual_augmentation_params,
    _to_teacher_image,
)
from .latent_loader import LatentSlotLoader

logger = get_logger(__name__)

_FALLBACK_CAPTION = "perform the demonstrated manipulation task"

# ── worker-side sample probe ────────────────────────────────────────────────
# RYNNVLA_SAMPLE_PROBE=1 turns this on. Off is the shipped behaviour: every timed
# block is skipped and `__getitem__` keeps its original one-line body.
#
# Why this lives HERE and not next to the trainer's `[stepprobe] DATALOADER WAIT`:
# that line times `next(epoch_iterator)` in the MAIN process, but the read happens in
# a DataLoader WORKER process, so the main process cannot see which sample the worker
# is chewing on. The 64-rank step probe already settled the LAYER -- step wall equals
# the slowest rank's data wait plus 1.01 s (n=21, spread 0.90-1.10), with resid 0.0 on
# all 1472 lines -- and ruled out compute, collectives, DeepSpeed opt, EMA and gen-2 GC.
# What is left is WHY one `next()` costs 10-46 s, and only the worker can answer that.
#
# It must also separate two explanations whose fixes are opposite:
#   one pathological sample taking 45 s     -> find and skip/downweight that sample
#   12 workers collectively falling behind  -> node-level I/O or CPU contention
# prefetch_factor=2 with num_workers=12 holds up to 24 ready batches, so a 45 s `next()`
# means all 24 drained -- which a threshold-only log cannot explain, since 100 samples at
# 1.5 s each would be silent. Hence the periodic AGG line carrying a histogram of EVERY
# sample plus the video/latent/proc/resid split, slow ones or not.
#
# `resid` is the discriminating field: it is the sample wall minus the three measured
# phases, i.e. augmentation, tensor allocation, index lookup, and above all any time the
# OS did not schedule this worker at all. resid dominating means contention, not I/O.
_SAMPLE_PROBE = os.environ.get("RYNNVLA_SAMPLE_PROBE", "") == "1"
# 8 s, matching the trainer's DATALOADER WAIT threshold, so a SLOW line and a main-process
# stall describe the same event and their counts can be compared directly. It only governs
# SLOW emission: the "all workers mildly behind" shape is carried by the AGG histogram over
# EVERY sample and is threshold-independent. At 2 s roughly 18% of healthy samples qualify,
# which works out to ~92k SLOW lines across 8 nodes and buries the ~23 real stalls.
_SAMPLE_PROBE_SLOW = float(os.environ.get("RYNNVLA_SAMPLE_PROBE_SECONDS", "8"))
# Kept small on purpose: 8 ranks x 12 workers share one node's sample stream, so a worker
# sees well under one sample per step and a cadence of 200 would emit nothing in a 300-step
# observation window.
_SAMPLE_PROBE_EVERY = max(1, int(os.environ.get("RYNNVLA_SAMPLE_PROBE_EVERY", "50")))
_SAMPLE_PROBE_EDGES = (0.5, 1.0, 2.0, 4.0, 8.0)

# Per-worker module state. Safe without a lock: ffmpeg's thread_count and torch's intra-op
# threads are native and never re-enter this module, so one worker touches this serially.
_sp_stage = {"video": 0.0, "latent": 0.0, "proc": 0.0, "worst": ("", "", 0.0)}
_sp_agg = {"n": 0, "sum": 0.0, "max": 0.0, "video": 0.0, "latent": 0.0, "proc": 0.0,
           "slow": 0, "slow_sum": 0.0, "hist": [0] * (len(_SAMPLE_PROBE_EDGES) + 1)}
# Keyed on pid, not a bool: DataLoader forks its workers, and a bool set by the parent
# before the fork would be inherited as already-done, silencing every child's WORKER line
# -- which is the line that proves worker log records reach the node log at all.
_sp_env_pid = None


def _sp_tag() -> str:
    info = get_worker_info()
    return f"r{os.environ.get('RANK', '?')}/w{-1 if info is None else info.id}"


def _sp_log_env_once() -> None:
    # Node-level CPU contention is one of the two live hypotheses, but 104 processes over
    # 84 physical cores is uniform across nodes, so it cannot alone explain why one node
    # produced 12 straggler events and another produced 0. The per-node affinity mask is
    # the missing variable and netcheck does not dump it. Emitting one line per worker
    # process also proves early -- inside the first batch -- that worker log records reach
    # the mirrored node log at all; if these never appear, the probe is silent rather than
    # "no slow samples", and those two must not be confused.
    global _sp_env_pid
    if _sp_env_pid == os.getpid():
        return
    _sp_env_pid = os.getpid()
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = -1
    logger.warning(
        f"[sampleprobe] WORKER tag={_sp_tag()} host={os.uname().nodename} pid={os.getpid()} "
        f"cpu_count={os.cpu_count()} affinity={affinity} "
        f"slow_at={_SAMPLE_PROBE_SLOW}s agg_every={_SAMPLE_PROBE_EVERY}"
    )


def _sp_record(ds, probe: int, total: float, attempt: int) -> None:
    _sp_log_env_once()
    agg = _sp_agg
    video, latent, proc = _sp_stage["video"], _sp_stage["latent"], _sp_stage["proc"]
    agg["n"] += 1
    agg["sum"] += total
    agg["max"] = max(agg["max"], total)
    agg["video"] += video
    agg["latent"] += latent
    agg["proc"] += proc
    bucket = len(_SAMPLE_PROBE_EDGES)
    for i, edge in enumerate(_SAMPLE_PROBE_EDGES):
        if total < edge:
            bucket = i
            break
    agg["hist"][bucket] += 1

    if total >= _SAMPLE_PROBE_SLOW:
        agg["slow"] += 1
        agg["slow_sum"] += total
        # Identity, not just duration: without the dataset/episode/path this line cannot
        # drive a fix. Capped like the read-failure logger above, because the AGG histogram
        # already carries the distribution and these lines are only for naming culprits.
        if agg["slow"] <= 30 or agg["slow"] % 100 == 0:
            path, kind, dt = _sp_stage["worst"]
            # Guarded for the same reason the read-failure logger guards its own
            # _probe_episode call below: _sp_record runs INSIDE __getitem__'s try, so an
            # exception raised here would be caught by the retry loop, blacklist a healthy
            # episode and silently train on a substitute sample. A broken probe must degrade
            # its own output, never the training data.
            try:
                ep = ds._probe_episode(probe)
                ident = f"ds={ds._ds_names[int(ds._ep_ds[ep])]} ep={ep}"
            except Exception as exc:
                ident = f"ds=? ep=? (identity lookup failed: {type(exc).__name__})"
            logger.warning(
                f"[sampleprobe] SLOW {total:.2f}s tag={_sp_tag()} index={probe} attempt={attempt} "
                f"{ident} "
                f"| video {video:.2f} latent {latent:.2f} proc {proc:.2f} "
                f"resid {max(0.0, total - video - latent - proc):.2f} "
                f"| worst {kind} {dt:.2f}s {path}"
            )

    if agg["n"] < _SAMPLE_PROBE_EVERY:
        return
    n = agg["n"]
    # resid is a derived remainder (sample wall minus the three measured phases), so it is
    # non-negative by construction; the clamp only suppresses float noise printing as -0.00.
    resid = max(0.0, agg["sum"] - agg["video"] - agg["latent"] - agg["proc"])
    logger.warning(
        f"[sampleprobe] AGG tag={_sp_tag()} n={n} mean={agg['sum']/n:.3f}s max={agg['max']:.2f}s "
        f"sum={agg['sum']:.1f}s | video {agg['video']:.1f} latent {agg['latent']:.1f} "
        f"proc {agg['proc']:.1f} resid {resid:.1f} "
        f"| slow>={_SAMPLE_PROBE_SLOW}s {agg['slow']} ({agg['slow_sum']:.1f}s = "
        f"{100.0*agg['slow_sum']/max(agg['sum'], 1e-9):.0f}% of read time) "
        f"| hist(<0.5,<1,<2,<4,<8,>=8)=" + ",".join(str(x) for x in agg["hist"])
    )
    agg.update({"n": 0, "sum": 0.0, "max": 0.0, "video": 0.0, "latent": 0.0, "proc": 0.0,
                "slow": 0, "slow_sum": 0.0})
    agg["hist"] = [0] * len(agg["hist"])


def _read_frame(video_path: str, index: int, view: Optional[str] = None) -> torch.Tensor:
    """Decode a single frame by stream-relative index, returning (H, W, C) uint8.

    Deliberately a single-threaded per-frame decode: a span-oriented frame-parallel
    decoder sets ``stream.thread_type = "FRAME"``, which pays for frame-parallel decoding
    of a *span*.
    Our access pattern is one random frame out of a 1080p/60fps file, where the thread
    spin-up measured 200-900 ms versus 11-70 ms single-threaded, and occasionally
    deadlocked (199 threads parked in futex_wait).
    """
    import av
    from .latent_sources import TarVideo, read_hdf5_frame, read_zarr_frame

    if index < 0:
        raise IndexError(f"Negative source frame: {index}")
    if video_path.lower().endswith((".hdf5", ".h5")):
        return torch.from_numpy(read_hdf5_frame(video_path, index, view))
    # A Zarr v3 episode store is a directory (EgoVerse). Checked before av.open, which
    # would raise IsADirectoryError. isdir is False for a missing path, so a bad video_path
    # still surfaces as FileNotFoundError from av.open rather than being misread as a store.
    if os.path.isdir(video_path):
        return torch.from_numpy(read_zarr_frame(video_path, index, view))
    member = TarVideo(video_path) if video_path.startswith("rovidx_tar://") else None
    try:
        container = av.open(member if member is not None else video_path)
    except BaseException:
        if member is not None:
            member.close()
        raise
    try:
        stream = container.streams.video[0]
        # Bound decoder threads per worker without serializing long-GOP seeks.
        stream.thread_count = 4
        rate = float(stream.average_rate)
        time_base = float(stream.time_base)
        start_pts = int(stream.start_time or 0)  # EPIC clips start at 50700, not 0
        container.seek(
            start_pts + int(round(index / rate / time_base)),
            stream=stream, any_frame=False, backward=True,
        )
        pos = -1
        for frame in container.decode(stream):
            if pos < 0:
                pos = int(round((frame.pts - start_pts) * time_base * rate)) if frame.pts is not None else 0
            else:
                pos += 1
            if pos >= index:
                return torch.from_numpy(frame.to_ndarray(format="rgb24"))
        raise IndexError(f"Could not decode frame {index} from {video_path}")
    finally:
        container.close()
        if member is not None:
            member.close()

# (dataset, manifest view name) -> VIEW_ROLES role, used only to give the views a canonical
# packing order (and to document what each camera physically is). Keyed by dataset because the
# same name means different things: EPIC's "main" is head-mounted, SthSthV2's "main" is
# third-person. RoboMIND's two side external cameras take the SEPARATE side_left/side_right
# roles -- they co-occur within episodes and would collide on a single `side`; sorted() is stable
# so their manifest order still breaks the tie deterministically. LEGACY: this (dataset, name)
# table only covers the 5-dataset pretrain-latent manifest, whose `views` is a LIST of
# {"view": raw_name}. The RynnVLA-Base manifests key `views` by an already-canonical camera
# name and are resolved through constants.VIEW_NAME_TO_ROLE instead.
_VIEW_ROLE = {
    ("EgoDex", "ego"): "head",
    ("EPIC_KITCHENS", "main"): "head",
    ("RDT-1B", "cam_high"): "head",
    ("SthSthV2", "main"): "front_third",
    ("RoboMind", "camera_top_rgb_images"): "head",
    ("RoboMind", "camera_left_rgb_images"): "left_wrist",
    ("RoboMind", "camera_right_rgb_images"): "right_wrist",
    ("RoboMind", "camera_front_rgb_images"): "front_third",
    ("RoboMind", "camera_front_external_rgb_images"): "front_third",
    ("RoboMind", "camera_left_external_rgb_images"): "side_left",
    ("RoboMind", "camera_right_external_rgb_images"): "side_right",
}

_INDEX_VERSION = 4
# v2 indexes are still readable: they predate windowed labeling, so every episode they describe
# was labeled from source frame 0 and an all-zero offset column is the exact reading, not a
# fallback. The frozen 2026-09-15 completed-subset trial snapshot is such an index.
# v3 is deliberately NOT accepted. It stored one start_frame per episode -- taken from views[0]
# -- and applied that single scalar to every view, but the views of one episode are concatenated
# into different files of very different lengths. Measured over the manifest: the per-view
# start_frame disagrees within an episode for 97.0% of Droid and 93.8% of BEHAVIOR-1K records.
# Where views[0]'s offset is too large the read raises IndexError (loud); where it is too small
# the frame decodes fine and is silently minutes away from the latents it is paired with. So a
# v3 fallback here would reintroduce the bug rather than degrade gracefully -- unlike v2, whose
# zeros are provably exact. v3 must fail at the version assert instead.
_ACCEPTED_INDEX_VERSIONS = (2, _INDEX_VERSION)


class _StringColumn:
    """Newline-separated UTF-8 blob + int64 offsets, appended to one string at a time.

    ``array``/``bytearray`` rather than a list of str: the full-corpus index holds ~45M
    strings (caption + video path + latent path per view), and a Python list of them costs
    ~36 bytes of overhead per element on top of the text. This keeps the same columns
    ``_unpack_strings`` reads, byte for byte.
    """

    def __init__(self) -> None:
        self.blob = bytearray()
        self.offsets = array("q", [0])

    def append(self, text: str) -> None:
        self.blob += text.encode("utf-8")
        self.blob += b"\n"
        self.offsets.append(len(self.blob))

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.blob:
            del self.blob[-1:]  # _unpack_strings expects a separator between, not after
        return np.frombuffer(self.blob, dtype=np.uint8), np.frombuffer(self.offsets, dtype=np.int64)


def _unpack_strings(blob: np.ndarray, offsets: np.ndarray) -> List[str]:
    raw = blob.tobytes()
    return [raw[int(a):int(b) - 1].decode("utf-8") for a, b in zip(offsets[:-1], offsets[1:])]


_PATH_MAP_ENV = "LATENT_PRETRAIN_PATH_MAP"


def _path_rewrites() -> List[Tuple[str, str]]:
    """Map index path prefixes across mounts using comma-separated src=dst rules."""
    rules: List[Tuple[str, str]] = []
    for entry in os.environ.get(_PATH_MAP_ENV, "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        src, sep, dst = entry.partition("=")
        if not sep or not src or not dst:
            raise ValueError(f"{_PATH_MAP_ENV} entry is not src=dst: {entry!r}")
        rules.append((src.rstrip("/"), dst.rstrip("/")))
    return rules


def build_index(
    manifest_path: str,
    out_path: str,
    latent_chunk: int = 6,
    latent_step_seconds: float = 0.25,
    default_fps: float = 30.0,
) -> Dict:
    """Flatten a manifest into a numpy index, dropping unusable episodes.

    An episode is dropped when it cannot supply one full chunk at its own stride
    (``num_latents < (chunk - 1) * stride + 1``): tail-clamping such a clip would emit a
    chunk of repeated latents, i.e. a constant target.

    Both manifest formats are streamed (``manifest_io``). A JSON-array manifest is sorted by
    ``(dataset, episode_id)`` first -- it is small enough to materialize, and the sort keeps
    indexes stable across manifest rewrites. A JSONL manifest is indexed in file order: the
    format exists for the ~15M-episode corpus, where materializing to sort would cost the
    tens of GB the streaming just avoided. ``meta['sorted']`` records which happened.
    """
    fmt = manifest_format(manifest_path)
    entries = iter_manifest(manifest_path, fmt)
    sorted_input = fmt == "array"
    if sorted_input:
        entries = list(entries)
        entries.sort(key=lambda e: (e["dataset"], str(e["episode_id"])))

    def _role_id(dataset: str, view_name: str) -> int:
        # Legacy 5-dataset manifests carry raw per-dataset camera names resolved through
        # _VIEW_ROLE; RynnVLA-Base manifests key views by an already-canonical camera name
        # (head / wrist_left / wrist_right / global / side) resolved through VIEW_NAME_TO_ROLE.
        role = _VIEW_ROLE.get((dataset, view_name))
        if role is None:
            role = VIEW_NAME_TO_ROLE.get(view_name)
        if role is None:
            raise KeyError(
                f"no camera-role mapping for view {view_name!r} of dataset {dataset!r}: "
                "add it to latent_pretrain._VIEW_ROLE or constants.VIEW_NAME_TO_ROLE"
            )
        return VIEW_ROLE_TO_ID[role]

    ds_names: List[str] = []
    ds_id: Dict[str, int] = {}
    # Columns are `array` buffers, not lists: a Python int costs ~28 bytes against the 2-8
    # its C typecode reserves, so five numeric columns per episode dominate the index once
    # the manifest is large. The typecodes are the C sizes the
    # np.frombuffer calls below assume (b=1, h=2, i=4, q=8, f=4).
    ep_ds, ep_nlat, ep_fps = array("h"), array("i"), array("f")
    ep_stride, ep_pair_stride = array("h"), array("b")
    ep_start_frame = array("i")
    view_off, view_role = array("q", [0]), array("b")
    # Per view, NOT per episode: the views of one episode are concatenated into different files
    # with different lengths, so each carries its own labeling-window offset. ep_start_frame above
    # is kept as views[0]'s offset for the smoke-data and prep-validation consumers.
    view_start_frame = array("i")
    captions, video_paths, latent_paths = _StringColumn(), _StringColumn(), _StringColumn()
    dropped: Dict[str, int] = {}
    kept: Dict[str, int] = {}

    for e in entries:
        name = e["dataset"]
        if name not in ds_id:
            ds_id[name] = len(ds_names)
            ds_names.append(name)
        views = sorted(e["views"], key=lambda v: _role_id(name, v["view"]))
        fps = float(views[0].get("fps") or 0.0) or default_fps
        # Latent i covers source frames (i*P, i*P+gap). P=1 is the frame-aligned LAM
        # extractor convention; P=4 is RynnLAM's stride-4 labeling. `stride` below is a
        # step in LATENT-INDEX space, so achieving a wall-clock step of
        # latent_step_seconds needs dividing by P -- without it a stride-4 corpus
        # trains on chunks 4x longer than intended and nothing reports it.
        pair_stride = max(1, int(views[0].get("pair_stride") or 1))
        stride = max(1, int(round(fps * latent_step_seconds / pair_stride)))
        # Some corpora were labeled as consecutive windows of one long recording, so latent 0
        # of such an episode is not source frame 0. The
        # offset has to travel with the index or the decoded image is minutes away from the
        # latents it is paired with -- and it has to travel PER VIEW, because each view was
        # concatenated into a different file at a different offset (in Droid and BEHAVIOR-1K
        # most manifest records have views whose start_frame disagrees). The
        # per-view value is written into view_start_frame in the loop below; this scalar is
        # views[0]'s offset, retained for callers that expect a per-episode scalar offset.
        start_frame = max(0, int(views[0].get("start_frame") or 0))
        nlat = min(int(v["num_latents"]) for v in views)
        if nlat < (latent_chunk - 1) * stride + 1:
            dropped[name] = dropped.get(name, 0) + 1
            continue
        kept[name] = kept.get(name, 0) + 1
        ep_ds.append(ds_id[name])
        ep_nlat.append(nlat)
        ep_fps.append(fps)
        ep_stride.append(stride)
        ep_pair_stride.append(pair_stride)
        ep_start_frame.append(start_frame)
        captions.append((e.get("caption") or "").strip() or _FALLBACK_CAPTION)
        for v in views:
            view_role.append(_role_id(name, v["view"]))
            view_start_frame.append(max(0, int(v.get("start_frame") or 0)))
            video_paths.append(v["video_path"])
            latent_paths.append(v["latent_path"])
        view_off.append(len(view_role))

    caption_blob, caption_off = captions.arrays()
    video_blob, video_pos = video_paths.arrays()
    latent_blob, latent_pos = latent_paths.arrays()
    meta = {
        "manifest": manifest_path,
        "latent_chunk": latent_chunk,
        "latent_step_seconds": latent_step_seconds,
        "sorted": sorted_input,
        "kept": kept,
        "dropped": dropped,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # np.savez writes a zip, which needs seek-on-write; the data lives on an OSS-fuse mount
    # that returns OSError(95) for that, so build locally and copy the finished file over.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = os.path.join(tmp, "index.npz")
        np.savez(
            tmp_path,
            version=np.asarray([_INDEX_VERSION]),
            meta=np.asarray([json.dumps(meta)]),
            ds_names=np.asarray(ds_names),
            ep_ds=np.frombuffer(ep_ds, dtype=np.int16),
            ep_nlat=np.frombuffer(ep_nlat, dtype=np.int32),
            ep_fps=np.frombuffer(ep_fps, dtype=np.float32),
            ep_stride=np.frombuffer(ep_stride, dtype=np.int16),
            ep_pair_stride=np.frombuffer(ep_pair_stride, dtype=np.int8),
            ep_start_frame=np.frombuffer(ep_start_frame, dtype=np.int32),
            view_off=np.frombuffer(view_off, dtype=np.int64),
            view_role=np.frombuffer(view_role, dtype=np.int8),
            view_start_frame=np.frombuffer(view_start_frame, dtype=np.int32),
            caption_blob=caption_blob, caption_off=caption_off,
            video_blob=video_blob, video_pos=video_pos,
            latent_blob=latent_blob, latent_pos=latent_pos,
        )
        shutil.copyfile(tmp_path, out_path)
    return meta


def _largest_remainder(exact: np.ndarray, total: int) -> np.ndarray:
    """Round ``exact`` to non-negative integers summing to EXACTLY ``total``.

    Exactness is load-bearing rather than tidy: the result sums to ``len(dataset)``, and
    ``max_steps`` is derived from that at submit time and sits in RESUME_CRITICAL_FIELDS, so a
    one-unit drift here silently voids every existing checkpoint of the arm. Largest remainder
    is used instead of per-element rounding precisely because per-element rounding does not
    conserve the total -- that off-by-one is what made ``sum(counts) != len(dataset)`` and cost
    the alpha=0.7 arm its locality shuffle.

    Ties break by position (``kind="stable"``) so every rank derives the same allocation from
    the same index without communicating.
    """
    floor = np.floor(exact).astype(np.int64)
    short = int(total) - int(floor.sum())
    # sum(exact) equals total only up to floating point, and the gap is not always negligible:
    # the weighted-mixing caller passes counts * (target / size), and for sizes where
    # size * (1 / size) lands below 1 -- 49 is the smallest -- a single-episode source bumped to
    # target 1 floors to 0 and leaves short == len(exact), which a half-open range check rejects
    # as a caller error and so crashes a valid configuration. Hence the tolerance check on the
    # proportions, which is what actually detects them not summing to total; a genuine error is
    # off by O(len(exact)) relative units, far outside it. The lower bound stays because a
    # negative short would make argsort[:short] silently drop a part instead of raising.
    if abs(float(exact.sum()) - total) > 1e-6 * max(1.0, abs(float(total))) or \
            not 0 <= short <= max(1, len(exact)):
        raise ValueError(
            f"_largest_remainder: cannot reach total {total} from {len(exact)} parts "
            f"(sum(exact)={float(exact.sum()):.6f}, floor sum={int(floor.sum())}); "
            "the requested proportions do not sum to total"
        )
    if short:
        remainder = exact - floor
        floor[np.argsort(-remainder, kind="stable")[:short]] += 1
    return floor


@DATASET_REGISTRY.register()
class LatentPretrainDataset(BaseVLADataset):
    """Stage-1 latent-action pretraining over the latent manifest.

    Args:
        data_path: the ``.npz`` index from ``build_index``; a raw ``vla_train.json`` also
            works but costs every rank a full JSON parse.
        latent_stats_path: latent mean/std json from ``scripts/build_latent_stats.py``.
        latent_chunk: latent steps per sample (6 downstream).
        latent_step_seconds: wall-clock span of one latent step; the per-episode stride is
            ``round(fps * this)``. 0.25 s matches LIBERO's 5 frames at 20 Hz.
        dataset_weights: ``{dataset_name: proportion}`` mixing weights; small sources get
            upsampled. Omit for natural proportions.
        max_retries: neighbouring samples to try when a video/latent read fails.
    """

    def __init__(
        self,
        *args,
        latent_stats_path: Optional[str] = None,
        latent_chunk: int = 6,
        latent_step_seconds: float = 0.25,
        latent_action_dim: int = 256,
        dataset_weights: Optional[Dict[str, float]] = None,
        max_retries: int = 8,
        **kwargs,
    ):
        data_path = args[0] if args else kwargs["data_path"]
        self._latent_chunk = int(latent_chunk)
        self._max_retries = int(max_retries)
        self._latent_loader = LatentSlotLoader(latent_action_dim, stats_path=latent_stats_path)
        self._blacklist = set()
        self._n_failures = 0

        if str(data_path).endswith(".npz"):
            index_path = data_path
        else:
            index_path = os.path.splitext(data_path)[0] + f"_index_c{latent_chunk}.npz"
            if not os.path.isfile(index_path):
                logger.warning(
                    f"No prebuilt index at {index_path}; parsing {data_path} in-process "
                    "(~90 s and 1.7 GB per rank). Run scripts/build_latent_pretrain_index.py once."
                )
                build_index(data_path, index_path, latent_chunk, latent_step_seconds)
        self._load_index(index_path)
        # The index freezes both the per-episode stride and the short-clip filter, so a
        # mismatched request would silently train at the wrong sampling rate.
        if int(self._index_meta["latent_chunk"]) != self._latent_chunk or \
                float(self._index_meta["latent_step_seconds"]) != float(latent_step_seconds):
            raise ValueError(
                f"index {index_path} was built with latent_chunk="
                f"{self._index_meta['latent_chunk']} / latent_step_seconds="
                f"{self._index_meta['latent_step_seconds']}, but this run asks for "
                f"{self._latent_chunk} / {latent_step_seconds}; rebuild it with "
                "scripts/build_latent_pretrain_index.py"
            )

        # One sample = one chunk start. Starts are spaced by half a chunk in wall-clock
        # terms, so consecutive indices are genuinely different samples (the base class's
        # frame-granular index space would hand back the same snapped sample many times).
        stride64 = self._ep_stride.astype(np.int64)
        self._start_stride = np.maximum(1, (stride64 * self._latent_chunk) // 2)
        span = (self._latent_chunk - 1) * stride64 + 1
        self._ep_starts = ((self._ep_nlat.astype(np.int64) - span) // self._start_stride) + 1

        super().__init__(*args, **kwargs)

        self._weight_slots = None
        self._weight_vbase = None
        self._weight_total = None
        if dataset_weights:
            self._build_weighted_index(dataset_weights)

        if latent_stats_path is None:
            logger.warning(
                "latent_stats_path is not set: latents will NOT be normalized (mean=0, std=1) "
                "and the flow-matching target is computed on raw latent values. For a real run "
                "point latent_stats_path at the output of scripts/build_latent_stats.py."
            )
        logger.info(
            f"LatentPretrainDataset: {len(self._ep_nlat)} episodes, {len(self)} samples, "
            f"chunk={self._latent_chunk}, latent step={latent_step_seconds}s "
            f"(stride {int(self._ep_stride.min())}..{int(self._ep_stride.max())} frames), "
            f"stats={'yes' if latent_stats_path else 'identity'}"
        )

    def _load_index(self, index_path: str):
        with np.load(index_path) as z:
            version = int(z["version"][0])
            assert version in _ACCEPTED_INDEX_VERSIONS, (
                f"stale index {index_path}: version {version}, this build reads "
                f"{_ACCEPTED_INDEX_VERSIONS} -- rebuild it with build_latent_pretrain_index.py"
            )
            self._index_meta = json.loads(str(z["meta"][0]))
            self._ds_names = [str(x) for x in z["ds_names"]]
            self._ep_ds = z["ep_ds"]
            self._ep_nlat = z["ep_nlat"]
            self._ep_fps = z["ep_fps"]
            self._ep_stride = z["ep_stride"]
            self._ep_pair_stride = z["ep_pair_stride"]
            # A v2 index has no windowed episodes, so zeros are exact rather than a guess.
            self._ep_start_frame = (z["ep_start_frame"] if "ep_start_frame" in z.files
                                    else np.zeros(len(self._ep_nlat), np.int32))
            self._view_off = z["view_off"]
            self._view_role = z["view_role"]
            # Same reasoning as ep_start_frame above: a v2 index predates windowed labeling, so
            # an all-zero per-view offset column is exact. There is no v3 fallback -- v3 stored
            # one offset per episode and is rejected by the version assert above, because
            # broadcasting it back out per view would reproduce the misalignment it caused.
            self._view_start_frame = (z["view_start_frame"] if "view_start_frame" in z.files
                                      else np.zeros(len(self._view_role), np.int32))
            self._captions = _unpack_strings(z["caption_blob"], z["caption_off"])
            self._video_paths = _unpack_strings(z["video_blob"], z["video_pos"])
            self._latent_paths = _unpack_strings(z["latent_blob"], z["latent_pos"])

        # The slot axis is the fixed VIEW_ROLES axis (see _build_sample), so every cached role
        # must index it. An index baked before the `top` -> `head` merge carries role 5; without
        # this check those episodes would raise inside _build_sample and be swallowed by the
        # __getitem__ retry loop -- a silent per-epoch sample drop. Fail at startup instead.
        _bad = np.unique(self._view_role[(self._view_role < 0)
                                         | (self._view_role >= NUM_VIEW_SLOTS)])
        if _bad.size:
            raise ValueError(
                f"{index_path} holds out-of-range camera roles {_bad.tolist()}; valid ids are "
                f"0..{NUM_VIEW_SLOTS - 1} ({VIEW_ROLES}). This index predates the current "
                "constants.VIEW_ROLES -- rebuild it with scripts/build_latent_pretrain_index.py."
            )

        rules = _path_rewrites()
        if rules:
            missing = [d for _, d in rules if not os.path.isdir(d)]
            if missing:
                raise FileNotFoundError(
                    f"{_PATH_MAP_ENV} rewrite target(s) not present on this host: {missing}"
                )
            rewritten = 0
            for i, p in enumerate(self._video_paths):
                for s, d in rules:
                    if p.startswith(s + "/"):
                        self._video_paths[i] = d + p[len(s):]
                        rewritten += 1
                        break
            logger.info(
                f"LatentPretrainDataset: rewrote {rewritten}/{len(self._video_paths)} video "
                f"paths via {len(rules)} ${_PATH_MAP_ENV} rule(s)"
            )
        logger.info(f"LatentPretrainDataset index {index_path}: {self._index_meta}")

    # ── weighted mixing ────────────────────────────────────────────────────────

    def _build_weighted_index(self, weights: Dict[str, float]):
        """Map a virtual index space onto per-EPISODE contiguous ranges so uniform sampling
        yields the requested proportions and the sampler's locality shuffle stays valid.

        The layout has to be episode-major. ``DistributedBatchSampler._locality_permutation``
        emits ``cumsum(counts)[e] + r``, i.e. it assumes index positions
        ``[base[e], base[e] + counts[e])`` all belong to episode ``e``, and
        ``_episode_sample_counts`` refuses -- silently falling back to a plain randperm --
        whenever the counts it is given do not sum to ``len(dataset)``. A dataset-major virtual
        layout (one contiguous segment per dataset, remapped inside by ``size // target``)
        satisfies neither condition: per-episode counts do not tile it, so virtual adjacency
        does not mean episode adjacency.

        Measured cost of that at alpha=0.7 on the 2B/64-GPU arm: 5.2x wall clock (0.841 ->
        4.13 s/step). The randperm destroyed the latent LRU's reuse, so nearly every sample
        re-read a whole npz to take 6 of its rows; slow-sample events went from 0.12/step to
        9.07/step and 14.7% of steps stalled 8-60s against 0.5% for the natural arm. In a
        synchronous run one rank's cold read stalls every rank at the gradient all-reduce,
        which the step probe books as ``fwd_bwd`` rather than ``data`` -- so rank0 looks
        compute-bound when it is actually waiting, and the misattribution sends you looking at
        the model instead of the sampler.

        Coverage is unchanged by the relayout, which is not obvious and was verified rather
        than assumed: per-dataset distinct real windows visited per epoch is identical under
        both layouts (``min(size, target)`` summed over datasets), and the per-episode spread
        is marginally tighter. It also stops whole episodes from being invisible: the
        dataset-major stride skipped a measurable number of episodes entirely, concentrated
        in the sources with the most windows per episode, whereas per-episode largest
        remainder gives every episode at least one slot at the same total cost.
        """
        missing = [n for n in self._ds_names if n not in weights]
        if missing:
            raise ValueError(f"dataset_weights is missing sources present in the index: {missing}")

        counts = self._ep_starts.astype(np.int64)
        total = int(self._cum_lengths[-1])
        wsum = sum(float(weights[n]) for n in self._ds_names)
        sizes = np.bincount(self._ep_ds.astype(np.int64), weights=counts,
                            minlength=len(self._ds_names)).astype(np.int64)

        # Largest remainder rather than per-element rounding: round() does not conserve the
        # total (up to +/- n/2 across n sources), and len(dataset) feeds max_steps at submit
        # time, which sits in RESUME_CRITICAL_FIELDS -- so an outer off-by-one here silently
        # voids every existing checkpoint of the arm, exactly as the inner one at line ~730
        # already guards against.
        exact = np.array([float(weights[name]) / wsum * total for name in self._ds_names])
        targets = _largest_remainder(exact, total)
        # max(1, ...) is carried over unchanged from the dataset-major layout: a zero weight
        # still buys one slot, so "exclude this source" cannot be expressed by weighting it 0.
        # Keeping it means the effective-rate table matches the old one. The bumps are counted
        # because they are the one reason the allocated total may legitimately exceed the
        # corpus, and the gate below compares against total + bumps rather than against a
        # value recomputed from the same array it is checking.
        bumps = int((targets < 1).sum())
        targets = np.maximum(targets, 1)

        slots = np.zeros(len(counts), dtype=np.int64)
        for di, name in enumerate(self._ds_names):
            target = int(targets[di])
            size = int(sizes[di])
            if size <= 0:
                continue
            eps = np.nonzero(self._ep_ds == di)[0]
            slots[eps] = _largest_remainder(counts[eps].astype(np.float64) * (target / size),
                                            target)

        vbase = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(slots, out=vbase[1:])
        weight_total = int(vbase[-1])
        # Hard gate, not a tidy-up: len(dataset) feeds max_steps at submit time and max_steps is
        # in RESUME_CRITICAL_FIELDS, so per-episode largest remainder must reproduce the corpus
        # size exactly -- plus one slot per bumped source -- or every existing checkpoint of
        # this arm is voided. Comparing against total + bumps catches both a non-conserving
        # outer rounding and a source that held a target but was skipped for having no windows.
        if weight_total != total + bumps:
            raise ValueError(
                f"weighted index total {weight_total} != corpus size {total} + {bumps} "
                "minimum-slot bump(s); len(dataset) would move and take max_steps with it"
            )

        self._weight_slots = slots
        self._weight_vbase = vbase
        self._weight_total = weight_total
        logger.info(
            "LatentPretrainDataset weighted mixing (episode-major): "
            + ", ".join(
                f"{name}={int(targets[di]) / weight_total:.3f}(raw {int(sizes[di]) / total:.3f})"
                for di, name in enumerate(self._ds_names)
            )
        )

    def _map_virtual_index(self, index: int) -> int:
        if self._weight_vbase is None:
            return index
        # Episode-major: vbase[e] <= index < vbase[e+1]. side="right" steps over any episode
        # allocated zero slots (vbase[e] == vbase[e+1]) so such an episode is never selected.
        e = int(np.searchsorted(self._weight_vbase, index, side="right")) - 1
        e = min(max(e, 0), len(self._weight_slots) - 1)
        cnt = int(self._ep_starts[e])
        slots = int(self._weight_slots[e])
        base = int(self._cum_lengths[e]) - cnt
        if slots <= 0:
            return base
        # Uniform stride INSIDE the episode, so a downsampled episode still contributes its
        # windows spread across its own duration rather than only from its start.
        return base + ((index - int(self._weight_vbase[e])) * cnt) // slots

    def __len__(self) -> int:
        if self._weight_vbase is not None:
            return self._weight_total
        return super().__len__()

    # ── index accessors ────────────────────────────────────────────────────────

    @property
    def episode_lengths(self) -> List[int]:
        """Per-episode size of the REAL index space.

        Must keep describing the real space even under a weighted index: ``base.py`` builds
        ``_cum_lengths`` from it, and ``_resolve_index`` / ``_target_to_source_index`` read it
        back per sample. The sampler wants the virtual space instead -- see
        ``sampler_slot_counts``.
        """
        return self._ep_starts.tolist()

    @property
    def sampler_slot_counts(self) -> List[int]:
        """Per-episode counts that tile the space the SAMPLER permutes.

        Equal to ``episode_lengths`` unless a weighted index is in effect, in which case it is
        the per-episode virtual slot allocation. ``DistributedBatchSampler`` prefers this and
        falls back to ``episode_lengths``, so every single-space dataset (all the Stage-2
        corpora) is unaffected.
        """
        if self._weight_slots is not None:
            return self._weight_slots.tolist()
        return self._ep_starts.tolist()

    def get_fps(self, episode_index: int) -> float:
        return float(self._ep_fps[episode_index])

    def get_robot_type(self, episode_index: int) -> RobotType:
        # Placeholder: this dataset never emits action/state, so the processor never looks
        # up a per-robot schema.
        return RobotType.FRANKA

    def _views(self, episode_index: int) -> List[Tuple[int, str, str, int]]:
        lo, hi = int(self._view_off[episode_index]), int(self._view_off[episode_index + 1])
        return [
            (int(self._view_role[i]), self._video_paths[i], self._latent_paths[i],
             int(self._view_start_frame[i]))
            for i in range(lo, hi)
        ]

    def load_instruction(self, episode_index: int, frame_index) -> List[str]:
        return [self._captions[episode_index]]

    def load_images(self, episode_index: int, frame_index) -> Dict[str, torch.Tensor]:
        start = frame_index if isinstance(frame_index, int) else int(frame_index[0])
        return {
            f"view{role}": self._read_view_frame(
                episode_index, role, video, view_start_frame, start).unsqueeze(0)
            for role, video, _, view_start_frame in self._views(episode_index)
        }

    def _read_view_frame(self, episode_index, role, video, start_frame, latent_start):
        """Decode the source frame that latent ``latent_start`` was extracted from.

        Latent i covers source frames ``(start_frame + i*pair_stride, ... + gap)``. Both parts
        of that mapping live here rather than at the call sites, because getting either wrong
        is invisible: the image simply comes from somewhere else in the recording than the
        latents it is paired with. ``start_frame`` is nonzero for the six windowed corpora, and
        it is PER VIEW: the views of one episode were concatenated into different files at
        different offsets (97.0% of Droid and 93.8% of BEHAVIOR-1K manifest records have views
        whose start_frame disagrees), so one episode-level scalar cannot serve all of them.
        """
        source_frame = (int(start_frame)
                        + latent_start * int(self._ep_pair_stride[episode_index]))
        # HDF5 and Zarr stores both hold several cameras behind one path, so both need the
        # labeler's source view recovered; a plain video file has exactly one stream.
        if not (video.lower().endswith((".hdf5", ".h5")) or os.path.isdir(video)):
            return _read_frame(video, source_frame)
        # The v2 index stores roles. Reverse the same injective mapping used by
        # build_index to recover the labeler's source view, never the packing order.
        dataset = self._ds_names[int(self._ep_ds[episode_index])]
        mapping = {name: r for (ds, name), r in _VIEW_ROLE.items() if ds == dataset}
        mapping = mapping or VIEW_NAME_TO_ROLE
        names = [name for name, r in mapping.items() if VIEW_ROLE_TO_ID[r] == role]
        if len(names) != 1:
            raise ValueError(f"Cannot recover source view for {dataset} role {role}: {names}")
        return _read_frame(video, source_frame, view=names[0])

    def load_action(self, episode_index: int, frame_index):
        raise NotImplementedError("LatentPretrainDataset has no robot actions (latent-only pretraining)")

    def load_state(self, episode_index: int, frame_index):
        raise NotImplementedError("LatentPretrainDataset has no robot states (latent-only pretraining)")

    def load_episode(self, episode_index: int, action_only: bool) -> Dict:
        raise NotImplementedError("LatentPretrainDataset has no per-episode action/state schema")

    def get_schema(self, num_workers: int = 8, process_group=None) -> Dict[str, Dict]:
        # No actions/states to normalize; an empty schema keeps both the single-dataset and
        # the ConcatDataset merge paths happy.
        return {"action": {}, "state": {}}

    # ── sampling ───────────────────────────────────────────────────────────────

    def _build_sample(self, index: int):
        if _SAMPLE_PROBE:
            _sp_stage["video"] = 0.0
            _sp_stage["latent"] = 0.0
            _sp_stage["proc"] = 0.0
            _sp_stage["worst"] = ("", "", 0.0)
        episode_index, ordinal, _ = self._resolve_index(self._map_virtual_index(index))
        if episode_index in self._blacklist:
            raise RuntimeError(f"blacklisted episode {episode_index}")

        stride = int(self._ep_stride[episode_index])
        # `start` indexes the LATENT array, not the video. _read_view_frame turns it into a
        # source frame (start_frame + start*pair_stride); doing that mapping there rather than
        # here keeps the two call sites from drifting apart.
        start = int(ordinal) * int(self._start_stride[episode_index])
        views = self._views(episode_index)

        # Image key "viewN" carries the role id purely so the processor's sorted-key image
        # order equals the latent row order below (single-digit roles sort as roles). The key
        # itself never reaches the prompt.
        images = {}
        # Fixed K = NUM_VIEW_SLOTS rows on the role axis, NOT len(views): row r belongs to
        # VIEW_ROLES[r] whether or not this episode owns that camera. Rows with no camera stay
        # zero and are key-masked via slot_mask, exactly as the expert's inactive-slot rule
        # expects (_build_action_stream_multiview), so a missing role is never read as a real
        # zero latent. Cost is NUM_VIEW_SLOTS rows instead of len(views) (<=3 in this corpus);
        # the tensor is tiny next to the frames.
        latents = torch.zeros(NUM_VIEW_SLOTS, self._latent_chunk, self._latent_loader.dim)
        slot_mask = torch.zeros(NUM_VIEW_SLOTS, dtype=torch.bool)
        if _SAMPLE_PROBE:
            _sp_t = time.monotonic()
        for role, video, latent_path, view_start_frame in views:
            # _load_index already range-checked the whole cache; this catches an index built
            # by an older writer that slipped past a resumed run.
            assert 0 <= role < NUM_VIEW_SLOTS, f"camera role {role} outside 0..{NUM_VIEW_SLOTS-1}"
            assert not slot_mask[role], (
                f"two cameras claim role {role} in episode {episode_index}; `views` is a dict "
                "keyed by camera name so this should be structurally impossible"
            )
            images[f"view{role}"] = self._read_view_frame(
                episode_index, role, video, view_start_frame, start)
            if _SAMPLE_PROBE:
                _dt = time.monotonic() - _sp_t
                _sp_stage["video"] += _dt
                if _dt > _sp_stage["worst"][2]:
                    _sp_stage["worst"] = (video, "video", _dt)
                _sp_t = time.monotonic()
            # NOTE: load_slots' `start_frame` is a LATENT index despite the name.
            chunk, _ = self._latent_loader.load_slots(
                {"one": latent_path}, start_frame=start, chunk=self._latent_chunk, stride=stride
            )
            if _SAMPLE_PROBE:
                _dt = time.monotonic() - _sp_t
                _sp_stage["latent"] += _dt
                if _dt > _sp_stage["worst"][2]:
                    _sp_stage["worst"] = (latent_path, "latent", _dt)
                # Reset here too, or the NEXT view's video timing starts before this view's
                # latent read and swallows it. That double-counts latent into video and also
                # biases `worst` toward video. Caught in the 17:00 run: video+latent summed to
                # 145% of the sample wall. total/latent/proc/resid were unaffected, so the run
                # stayed usable via video_true = total - latent - proc - resid.
                _sp_t = time.monotonic()
            latents[role] = chunk[0]
            slot_mask[role] = True

        # SF/LB teacher input: un-augmented frames, sorted-key order (== slot order here) to
        # match the processor's image layout and image_grid_thw row order.
        teacher_images = None
        if self.emit_teacher_images:
            teacher_images = torch.stack([_to_teacher_image(images[k]) for k in sorted(images)])

        if self.visual_augmentation:
            aug_params = _sample_visual_augmentation_params()
            images = {k: _apply_visual_augmentation(v, aug_params) for k, v in images.items()}

        outputs = {
            "robot_type": self.get_robot_type(episode_index),
            "text": self._captions[episode_index],
            "images": images,
            "latent_targets": latents,
            # (K,) on the role axis, so it lines up row-for-row with latent_targets above and
            # with the seeds _pool_view_seeds scatters by camera_slot_ids. Fixed width also
            # makes the collator's pad-to-batch-max a no-op instead of a ragged join.
            "slot_mask": slot_mask,
            # Role ids, NOT sample order. This is the whole point of the fix: it is the same
            # axis stage-2 emits (LiberoPlusDataset.camera_slot_map -> front_third=3,
            # left_wrist=1), so slot_seed_proj / view_role_emb / the expert's per-slot
            # attention all mean the same thing in pretraining and in fine-tuning. Passing the
            # map at all is what makes the processor emit camera_slot_ids.
            "camera_slot_map": {f"view{role}": role for role, _, _, _ in views},
        }
        if self.processor is None:
            if teacher_images is not None:
                outputs["teacher_images"] = teacher_images
                outputs["primary_teacher_index"] = torch.tensor(0, dtype=torch.long)
            return outputs
        if _SAMPLE_PROBE:
            _sp_t = time.monotonic()
        features = self.processor(**outputs, return_tensors="pt")
        if _SAMPLE_PROBE:
            _sp_stage["proc"] += time.monotonic() - _sp_t
        if getattr(self.processor, "use_state", False):
            action_dim = int(self.processor.get_config_overrides()["action_dim"])
            features["states"] = torch.zeros(1, 1, action_dim, dtype=torch.float32)
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
            features["primary_teacher_index"] = torch.tensor(0, dtype=torch.long)
        return features

    def __getitem__(self, index: int):
        n = len(self)
        for attempt in range(self._max_retries):
            probe = (index + attempt * 9973) % n
            try:
                if _SAMPLE_PROBE:
                    # Timed around the successful build only. A failed attempt is already
                    # reported by the read-failure logger below, and charging its partial
                    # work to the read-time distribution would mix two different questions.
                    _sp_t0 = time.monotonic()
                    sample = self._build_sample(probe)
                    _sp_record(self, probe, time.monotonic() - _sp_t0, attempt)
                    return sample
                return self._build_sample(probe)
            except Exception as exc:  # unreadable video / latent npz / short clip
                # The asserts inside _build_sample are corpus invariants (camera role in range,
                # no two cameras claiming one role), not read failures. Retrying past them would
                # relabel a broken index as "unreadable video" and quietly train on a substitute
                # sample, so they must crash instead of being absorbed here.
                if isinstance(exc, AssertionError):
                    raise
                self._n_failures += 1
                try:
                    ep = self._probe_episode(probe)
                    self._blacklist.add(ep)
                    where = f"{self._ds_names[int(self._ep_ds[ep])]} episode {ep}"
                except Exception:
                    where = f"index {probe}"
                if self._n_failures <= 20 or self._n_failures % 500 == 0:
                    logger.warning(
                        f"LatentPretrainDataset: skipping {where} "
                        f"({type(exc).__name__}: {exc}); total failures {self._n_failures}"
                    )
        raise RuntimeError(
            f"LatentPretrainDataset: {self._max_retries} consecutive read failures around index {index}"
        )

    def _probe_episode(self, index: int) -> int:
        return int(np.searchsorted(self._cum_lengths, self._map_virtual_index(index), side="right"))
