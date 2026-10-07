#!/usr/bin/env python3
"""Rebuild LatentPretrainDataset manifests from source JSONs + the latent files on disk.

Why this exists instead of reading RynnLAM's ``index_shard*.json``: the dense labeler
built ``index`` as a fresh list per python invocation but wrote it to a path shared
by every dataset of that shard, atomically overwriting it -- and the sweep
invoked python once per dataset. Each dataset therefore erased the entries of the dataset
before it, so a finished shard's index held only its LAST dataset. Measured on the
2026-09-16 snapshot: 1,000,533 of ~15.25M labeled episodes (6.6%) survived. The latent
``.npz`` files are per-episode and published atomically, so they are intact; only the index
is lost, and this script rebuilds the manifest from what is still authoritative:

  * ``<data-root>/<Dataset>.json``  dataset / episode_id / caption / views.video_path
  * ``<latent-dir>/<Dataset>/<episode_id>/<view>/latent.npz`` -> its ``meta`` member carries
    num_latents, code_shape, fps and protocol.{pair_stride,gap,start_frame,source}, i.e.
    every field ``latent_pretrain.build_index`` needs, read from the file the labeler wrote.

Output is JSONL, one episode per line: a full-corpus manifest is gigabytes of JSON and
``json.load`` on it costs roughly six times that in RSS (latent_pretrain.py's docstring
documents the measured ratio). Per-(dataset, shard) part files give crash-safe resume -- a part is
published with ``os.replace`` only once its dataset finished -- and ``--combine`` concatenates
them byte-wise into the single manifest that the Stage-1 preparation path feeds to
build_index and build_latent_stats.

Cost, measured on a network object-store FUSE mount 2026-09-16, tail read below:
~460-650 files/s PER NODE, and the cap does not multiply with processes -- 1 process x 32
threads measured 461 files/s while 4 processes x 32 threads measured 441 files/s *in aggregate*
(120/s each). Per-client round-trip latency is the limit, not CPU, so a second job on the same
node only steals from the first. 32 threads is the knee: 8 threads 181/s, 64 threads 646/s with
p95 352 ms, 128 threads 630/s with p95 851 ms. Overload is worse than slow -- 256 in-flight
requests (4 x 64) drove single lookups past 25 s with every worker parked in D state.
A full-corpus sweep is therefore hours from ONE node but tens of minutes per shard, so shard
across nodes (``--shard i --num-shards N``, the pattern RynnLAM's own labeling jobs use)
rather than across processes. Sharding is by position in the source JSON, so
a shard's slice is stable across reruns.
Second, separate cost: every shard streams the whole source JSON (it parses every record
and keeps 1/N), which is minutes from a local copy and hours from the mount at ~2 MB/s.

Usage:
    cd <repo-root>
    # one dataset, single process:
    python3 scripts/rebuild_latent_manifest.py --datasets <DatasetName>
    # one shard of a parallel sweep (submit N of these, i = 0..N-1):
    python3 scripts/rebuild_latent_manifest.py --shard 0 --num-shards 32
    # once every shard finished, concatenate the parts into one manifest:
    python3 scripts/rebuild_latent_manifest.py --num-shards 32 --expect-shards 32 --combine-only
"""

import argparse
import ast
import fnmatch
import json
import os
import struct
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Deliberately no import of rynnvla.datasets.*: its __init__ pulls in torch, and this script
# runs on CPU-only nodes next to the labeling jobs.
from rynnvla.constants import VIEW_NAME_TO_ROLE, VIEW_ROLE_TO_ID  # noqa: E402
from rynnvla.utils.manifest_io import iter_json_array, join_caption  # noqa: E402

_ROOT = os.environ.get("RYNNVLA_DATA_ROOT", "")
DEFAULT_DATA_ROOT = f"{_ROOT}/RynnVLA-Base" if _ROOT else ""
DEFAULT_LATENT_DIR = f"{_ROOT}/RynnVLA-Base-latent-ktoken608/latents" if _ROOT else ""
DEFAULT_OUT_DIR = f"{_ROOT}/pretrain-latent-rynnvla-base" if _ROOT else ""
# The labeling sweep skipped these in the data root; mirroring that keeps the rebuilt
# corpus identical to the labeled one instead of adding caption/backup artifacts.
NON_DATASET_PATTERNS = (
    "caption*", "nocaption*", "*_result", "remapped", "original_backup",
    "build_*", "rebuild_*", "remap_*", "*.bak",
)
# Corpora whose source is not a video container, so RynnLAM's FrameReader never learned an
# fps and wrote ``fps: 0`` into the latent meta (measured across the corpus: RoboMIND2.0 HDF5
# and EgoVerse zarr are the only two). Leaving a zero would make
# build_index substitute its --default-fps: for RoboMIND2.0 that is 30 instead of the measured
# ~14, so stride becomes round(30*0.25/4)=2 instead of round(14*0.25/4)=1 and one chunk spans
# 5*2*4/14 = 2.86 s of video instead of 5*1*4/14 = 1.43 s -- every chunk 2x too long, with
# nothing downstream to report it. So the fps is resolved here and counted. HDF5 episodes are
# measured from their own timestamps (``hdf5_fps``), which succeeds for essentially every view;
# this table is only the fallback for when that read fails.
#
# These are last-resort guesses, NOT per-dataset constants -- fps genuinely varies per episode.
# Measured over a large per-dataset sample: HowTo100M has 562 distinct values
# (29.97 x56.7%, 30 x16.0%, 25 x11.7%, 23.976 x10.4%) and RoboMIND2.0 has 1096. The real value
# always comes from the npz meta. These have never fired (no `fps_fallback` counter has ever
# been non-zero).
STATIC_FPS = {
    "EPIC_KITCHENS": 59.94,  # bimodal in reality: 50 x54.4% / 59.94 x44.4% -- a guess either way
    "SthSthV2": 12.0,
    "RoboMind": 20.0,
    # Median of those 1096 measured values (14 x2.7%, 13 x1.6%, 15 x1.3%, 14.5 x1.0%, ...).
    # An earlier single-file hand check gave 101 Hz (2225 frames / 22 s) and that outlier was
    # coded here; it is 7x too high and would have made stride 6 instead of 1.
    "RoboMIND2.0": 14.0,
    "EgoDex": 30.0,
    "EgoVerse": 30.0,  # zarr.json attributes.fps
    "RDT-1B": 30.0,
}
# What latent_pretrain._read_frame can decode. A source outside this set cannot produce a
# training image, so its episodes are left out of the manifest rather than indexed and then
# skipped sample by sample by the __getitem__ retry loop (the silent drop that
# latent_pretrain.py:421 exists to prevent).
VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".mpg", ".mpeg")


def source_kind(path: str) -> str:
    """'tar', 'hdf5', 'video', 'zarr' or 'unsupported' -- the backends _read_frame dispatches on.

    'zarr' is a candidate, not a confirmation: a suffix-less path is either a Zarr v3 store
    directory (EgoVerse, whose ``video_path`` is the episode directory) or a mistake, and
    telling those apart costs one stat that the caller performs. Implemented here rather than
    imported from latent_sources because that module pulls in PIL, and this script is
    deliberately torch/PIL-free so it runs on any interpreter with numpy.
    """
    if path.startswith("rovidx_tar://"):
        return "tar"
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".hdf5", ".h5"):
        return "hdf5"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if not suffix:
        return "zarr"
    return "unsupported"


def is_zarr_store(path: str) -> bool:
    """One stat pair: does this directory carry Zarr v3 (or legacy v2) array metadata?"""
    return (os.path.isfile(os.path.join(path, "zarr.json"))
            or os.path.isfile(os.path.join(path, ".zarray")))


def hdf5_fps(path: str) -> Optional[float]:
    """Frame rate of a RoboMIND-style trajectory, from its own integer-second timestamps.

    ``camera_observations/timestamp`` has one entry per stored frame, so (n-1)/span is the
    average capture rate. The stamps are whole seconds, which quantises a 22 s episode to
    about +-5% -- far tighter than the stride needs, since stride = round(fps*0.25/pair_stride)
    only moves at a 12.5% change for this corpus.
    """
    try:
        import h5py

        with h5py.File(path, "r", locking=False) as f:
            stamps = f["camera_observations/timestamp"]
            count = stamps.shape[0]
            if count < 2:
                return None
            span = float(stamps[count - 1]) - float(stamps[0])
        return (count - 1) / span if span > 0 else None
    except Exception:
        return None


# ── npz meta reader ───────────────────────────────────────────────────────────

def _npy_header(buf: bytes, off: int) -> Tuple[dict, int]:
    """Parse the .npy header at ``off``; return (header dict, payload offset)."""
    if buf[off:off + 6] != b"\x93NUMPY":
        raise ValueError(f"bad .npy magic at {off}: {buf[off:off + 6]!r}")
    major = buf[off + 6]
    if major == 1:
        length, start = struct.unpack_from("<H", buf, off + 8)[0], off + 10
    elif major in (2, 3):
        length, start = struct.unpack_from("<I", buf, off + 8)[0], off + 12
    else:
        raise ValueError(f"unsupported .npy version {major}")
    # numpy writes this header as a python literal: {'descr': ..., 'fortran_order': ..., 'shape': ...}
    return ast.literal_eval(buf[start:start + length].decode("latin1")), start + length


def read_npz_meta(path: str, tail_bytes: int = 1 << 16) -> dict:
    """The ``meta`` member of a latent .npz, from a single read of the file's tail.

    ``np.savez`` stores members uncompressed in insertion order and RynnLAM writes ``meta``
    last, so it sits just before the zip central directory: one tail read yields the whole
    labeling protocol. That is one round trip instead of the three ``np.load`` needs (TOC at
    EOF, member header at BOF, member payload) -- the entire cost on a latency-bound mount.
    """
    # One open + one read, no os.path.getsize: on a latency-bound mount the stat is a round
    # trip of its own, and SEEK_END makes it unnecessary.
    with open(path, "rb") as f:
        try:
            f.seek(-tail_bytes, os.SEEK_END)
        except OSError:
            f.seek(0)  # file shorter than the tail window: read all of it
        tail = f.read()
    idx = tail.rfind(b"PK\x03\x04")
    while idx != -1:
        name_len = struct.unpack_from("<H", tail, idx + 26)[0]
        extra_len = struct.unpack_from("<H", tail, idx + 28)[0]
        if tail[idx + 30:idx + 30 + name_len] == b"meta.npy":
            header, off = _npy_header(tail, idx + 30 + name_len + extra_len)
            dtype = np.dtype(header["descr"])
            count = int(np.prod(header["shape"])) if header["shape"] else 1
            raw = tail[off:off + count * dtype.itemsize]
            if len(raw) != count * dtype.itemsize:
                raise ValueError(f"meta member truncated: {len(raw)} bytes in tail")
            # numpy stores <U as UCS4, so the payload is utf-32-le rather than utf-8.
            return json.loads(raw.decode("utf-32-le" if dtype.kind == "U" else "utf-8"))
        idx = tail.rfind(b"PK\x03\x04", 0, idx)
    raise ValueError("meta.npy local header not inside the tail read")


def read_meta_fallback(path: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if "meta" not in archive.files:
            raise ValueError("npz has no meta member")
        return json.loads(str(archive["meta"]))


def probe_latent(path: str, tail_bytes: int) -> Tuple[Optional[dict], str]:
    """(meta, how): how is 'tail', 'npz' (fast path failed), or a skip reason."""
    try:
        return read_npz_meta(path, tail_bytes), "tail"
    except FileNotFoundError:
        return None, "missing_latent"
    except Exception:
        pass
    try:
        return read_meta_fallback(path), "npz"
    except FileNotFoundError:
        return None, "missing_latent"
    except Exception:
        return None, "unreadable_latent"


# ── naming and dataset discovery ──────────────────────────────────────────────

def is_safe_name(value: str, *, nested: bool) -> bool:
    """The labeler's ``safe_name`` rule, reimplemented (it raises; this reports).

    The labeler refused to write latents under names that fail it, so such episodes are
    simply unlabeled -- but rebuilding their paths without the same check would let an
    episode_id like ``../other`` point outside the latent root.
    """
    if not value or value in {".", ".."} or "\\" in value:
        return False
    parts = value.split("/")
    if value.startswith("/") or ".." in parts:
        return False
    return nested or len(parts) == 1


def is_dataset_json(path: str) -> bool:
    if not path.endswith(".json"):
        return False
    stem = os.path.basename(path)[:-5]
    if any(fnmatch.fnmatch(stem, pattern) for pattern in NON_DATASET_PATTERNS):
        return False
    # A dataset JSON is a list of {dataset, episode_id, views}; check the header only,
    # because validating a multi-GB file by parsing it is what the labeler avoids too.
    try:
        with open(path, "rb") as f:
            return b'"views"' in f.read(4096)
    except OSError:
        return False


def discover_datasets(data_root: str, wanted: Optional[List[str]]) -> Dict[str, str]:
    files = {name[:-5]: os.path.join(data_root, name)
             for name in sorted(os.listdir(data_root))
             if is_dataset_json(os.path.join(data_root, name))}
    if wanted:
        missing = [d for d in wanted if d not in files]
        if missing:
            raise SystemExit(f"not a labelable dataset JSON under {data_root}: {missing}\n"
                             f"available: {sorted(files)}")
        return {d: files[d] for d in wanted}
    return files


def role_id(view_name: str) -> Optional[int]:
    """Camera-role id for a canonical RynnVLA-Base view name, or None when unmapped.

    Only constants.VIEW_NAME_TO_ROLE is consulted, not latent_pretrain._VIEW_ROLE: RynnVLA-Base
    JSONs already key views canonically (head / wrist_left / wrist_right / global / side --
    verified across all 32 labelable datasets), and _VIEW_ROLE's legacy per-dataset names
    belong to the old pretrain-latent manifest. An unmapped name here is fatal, loudly, rather
    than a KeyError inside build_index later.
    """
    role = VIEW_NAME_TO_ROLE.get(view_name)
    return VIEW_ROLE_TO_ID[role] if role else None


# ── per-episode rebuild ───────────────────────────────────────────────────────

class Options:
    """Read-only probe/emit settings shared by every worker thread."""

    def __init__(self, args):
        self.latent_dir = args.latent_dir
        self.tail_bytes = args.tail_bytes
        self.max_caption_chars = args.max_caption_chars
        self.latent_dim = args.expected_latent_dim
        self.pair_stride = args.expected_pair_stride
        self.representation = args.expected_representation
        self.default_fps = args.default_fps


def rebuild_episode(item: dict, opt: Options) -> Tuple[Optional[dict], List[str]]:
    """One source-JSON episode -> (manifest line, counter keys); line is None when dropped.

    Counter keys come back to the caller instead of being incremented here: ``Counter[k] += 1``
    from 64 threads loses updates, and these numbers are the only evidence of why the rebuilt
    corpus is smaller than the labeled one.

    Every view must have a latent. The labeler keeps a partially-labeled episode in its index,
    but training one silently drops a camera, so those are counted and left out.
    """
    keys: List[str] = []
    dataset, episode, views = item.get("dataset"), item.get("episode_id"), item.get("views")
    if not dataset or episode is None or not isinstance(views, dict) or not views:
        return None, ["malformed"]
    dataset, episode = str(dataset), str(episode)
    if not is_safe_name(dataset, nested=False) or not is_safe_name(episode, nested=True):
        return None, ["unsafe_name"]

    unsafe = [view for view in views if not is_safe_name(view, nested=False)]
    if unsafe:
        return None, ["unsafe_name"]
    roles = {view: role_id(view) for view in views}
    unmapped = [view for view, role in roles.items() if role is None]
    if unmapped:
        return None, [f"unmapped_camera:{view}" for view in unmapped]

    out_views, incomplete, gap, latent_dim = [], False, None, None
    for view in sorted(views, key=lambda v: roles[v]):
        source = views[view]
        if not isinstance(source, dict) or not source.get("video_path"):
            return None, ["no_video_path"]
        # Checked before the latent probe: an undecodable source makes the episode unusable
        # no matter what its latents say, and skipping the probe saves a round trip.
        kind = source_kind(str(source["video_path"]))
        if kind == "zarr":
            # One stat, and only for suffix-less paths (EgoVerse zarr stores). A
            # directory that is not a Zarr store cannot produce a training image either, and
            # waving it through would move the failure into __getitem__ thousands of steps
            # into a run -- the silent drop this whole check exists to prevent.
            if not is_zarr_store(str(source["video_path"])):
                return None, ["unsupported_source:dir"]
        elif kind == "unsupported":
            return None, [f"unsupported_source:{os.path.splitext(str(source['video_path']))[1] or 'dir'}"]
        path = os.path.join(opt.latent_dir, dataset, episode, view, "latent.npz")
        meta, how = probe_latent(path, opt.tail_bytes)
        keys.append(f"probe_{how}")
        if meta is None:
            incomplete = True
            keys.append(how)
            continue
        protocol = meta.get("protocol") or {}
        shape = meta.get("code_shape") or []
        num_latents = meta.get("num_latents")
        if not num_latents or int(num_latents) < 1:
            return None, ["bad_num_latents"]
        if len(shape) != 1 or (opt.latent_dim and int(shape[0]) != opt.latent_dim):
            return None, [f"latent_dim_mismatch:{shape}"]
        # Latent i covers source frames (start_frame + i*P, ... + gap). Several corpora were
        # labeled as consecutive windows of one long recording, so start_frame is nonzero for
        # a substantial minority of episodes; it travels in the manifest and the index so the
        # decoded frame lands on the latents it is paired with. It has to travel PER VIEW
        # (index v4 view_start_frame): each view was concatenated into a different file at a
        # different offset, so one episode-level scalar cannot serve all of them.
        start_frame = int(protocol.get("start_frame") or 0)
        if start_frame < 0:
            return None, [f"negative_start_frame:{start_frame}"]
        if start_frame:
            keys.append(f"windowed_start_frame:{dataset}")
        stride = int(protocol.get("pair_stride") or 1)
        if opt.pair_stride and stride != opt.pair_stride:
            return None, [f"pair_stride_mismatch:{stride}"]
        if opt.representation and protocol.get("representation") != opt.representation:
            return None, [f"representation_mismatch:{protocol.get('representation')}"]
        if gap is None:
            gap = int(protocol.get("gap") or 0) or int(item.get("gap") or 5)
        latent_dim = int(shape[0])
        # protocol.source is the path the labeler itself opened (resolve_source: symlinks
        # resolved, rovidx_tar:// URLs kept verbatim), i.e. proven readable.
        video = protocol.get("source") or source["video_path"]
        keys.append("video_path_resolved" if video != source["video_path"] else "video_path_verbatim")
        if not protocol.get("source"):
            keys.append("no_protocol_source")
        fps = float(meta.get("fps") or 0.0)
        if fps <= 0:
            # The labeler could not probe a frame rate (HDF5/zarr sources), so build_index
            # would substitute its own --default-fps and silently rescale every chunk of this
            # corpus. Measure it from the source where that is possible, else use the table.
            measured = hdf5_fps(str(video)) if kind == "hdf5" else None
            if measured:
                fps = float(measured)
                keys.append(f"fps_measured_hdf5:{dataset}")
            else:
                fps = float(STATIC_FPS.get(dataset) or opt.default_fps)
                keys.append(f"fps_fallback:{dataset}:{fps:g}")
        out_views.append({
            "view": view,
            "video_path": video,
            "latent_path": path,
            "num_latents": int(num_latents),
            "fps": fps,
            "pair_stride": stride,
            "start_frame": start_frame,
        })
    if incomplete:
        return None, keys + ["incomplete_views"]
    if not out_views:
        return None, keys + ["no_usable_views"]
    entry = {
        "dataset": dataset,
        "episode_id": episode,
        "caption": join_caption(item.get("caption"), opt.max_caption_chars),
        "gap": gap if gap is not None else int(item.get("gap") or 5),
        "latent_dim": latent_dim or opt.latent_dim,
        "views": out_views,
    }
    return entry, keys + [f"views_{len(out_views)}", "kept"]


def part_path(out_dir: str, dataset: str, shard: int, num_shards: int) -> str:
    """num_shards is encoded in the name: combining parts from two different shardings would
    duplicate episodes in the manifest, and nothing downstream would notice."""
    return os.path.join(out_dir, "manifests", "parts",
                        f"{dataset}.n{num_shards}.shard{shard:04d}.jsonl")


def rebuild_dataset(dataset: str, path: str, out_dir: str, args, opt: Options,
                    executor: ThreadPoolExecutor) -> dict:
    """Stream one dataset JSON, publish its part file, return counters."""
    destination = part_path(out_dir, dataset, args.shard, args.num_shards)
    if os.path.isfile(destination) and not args.force:
        with open(destination, "rb") as f:
            reused = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 22), b""))
        return {"dataset": dataset, "kept": reused, "reused": True,
                "counters": {"reused": reused}, "skips": {}, "part": destination, "seconds": 0.0}

    counters: Counter = Counter()
    # Source JSONs can repeat an episode_id, and a duplicate would be sampled twice. This is
    # the one per-dataset structure held in memory: ~9M ids for HowTo100M on shard 0.
    seen = set()
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    started = time.perf_counter()
    tmp = tempfile.NamedTemporaryFile("w", dir=os.path.dirname(destination), prefix=f".{dataset}.",
                                      suffix=".jsonl", delete=False, encoding="utf-8")
    kept = scanned = 0
    try:
        batch: List[dict] = []

        def flush(items: List[dict]):
            nonlocal kept
            if not items:
                return
            for entry, keys in executor.map(lambda e: rebuild_episode(e, opt), items):
                counters.update(keys)
                if entry is None:
                    continue
                if entry["episode_id"] in seen:
                    counters["duplicate_episode"] += 1
                    continue
                seen.add(entry["episode_id"])
                tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept += 1

        for item in iter_json_array(path):
            scanned += 1
            if args.num_shards > 1 and (scanned - 1) % args.num_shards != args.shard:
                continue
            batch.append(item)
            if len(batch) >= 256:
                flush(batch)
                batch = []
            if args.limit and kept >= args.limit:
                counters["capped"] += 1
                break
        flush(batch)
    finally:
        tmp.close()
    seconds = time.perf_counter() - started
    if scanned:
        # Publish even when nothing was kept: an empty part means "this slice was scanned and
        # no episode survived", which is a completed state, and --expect-shards needs to tell
        # it apart from "this shard never ran" (no file at all). Without it, a small dataset
        # whose 1/N slice holds no labeled episode would block the whole combine. Atomicity
        # still holds -- a killed run leaves no file, so a resume redoes the slice.
        os.replace(tmp.name, destination)
    else:
        # The source JSON yielded nothing (empty array, or unreadable): no scan happened, so
        # no part may claim one. A part left over from an earlier run has to go, because it
        # now describes episodes this run could not confirm.
        os.unlink(tmp.name)
        if os.path.isfile(destination):
            os.unlink(destination)
    probes = sum(v for k, v in counters.items() if k.startswith("probe_"))
    return {"dataset": dataset, "kept": kept, "scanned": scanned, "reused": False,
            "counters": dict(counters),
            "skips": {k: v for k, v in counters.most_common()
                      if not k.startswith(("probe_", "views_", "kept", "video_path_",
                                           "windowed_start_frame", "fps_measured"))},
            "part": destination if kept else None, "seconds": round(seconds, 1),
            "files_per_second": round(probes / seconds, 1) if seconds and probes else None}


# ── combine / report ──────────────────────────────────────────────────────────

def _part_shard(path: str) -> int:
    return int(os.path.basename(path).rsplit("shard", 1)[1][:4])


def combine_parts(out_dir: str, name: str, num_shards: int, expect_shards: int) -> Optional[dict]:
    """Concatenate this sharding's part files byte-wise into manifests/<name>.jsonl.

    Byte-wise rather than JSON-wise: the combined manifest is gigabytes, and parsing it just to
    re-emit it would need the memory this format exists to avoid.
    """
    parts_dir = os.path.join(out_dir, "manifests", "parts")
    if not os.path.isdir(parts_dir):
        return None
    suffix = f".n{num_shards}.shard"
    groups: Dict[str, List[str]] = {}
    for entry in sorted(os.listdir(parts_dir)):
        if suffix in entry and entry.endswith(".jsonl"):
            groups.setdefault(entry.split(suffix)[0], []).append(os.path.join(parts_dir, entry))
    if not groups:
        return None
    missing = {ds: sorted(set(range(expect_shards)) - {_part_shard(p) for p in paths})
               for ds, paths in groups.items()} if expect_shards else {}
    incomplete = {ds: m for ds, m in missing.items() if m}
    if incomplete:
        # Refuse BEFORE writing: an incomplete manifest at the path preparation reads looks
        # exactly like a complete one, and every downstream consumer (build_index, stats)
        # would happily train on the fraction that happened to be there. This fired for real
        # on 2026-09-16 when a fuse disconnect killed 28 of 32 shards mid-sweep.
        raise SystemExit(f"[combine] FATAL: {len(incomplete)} dataset(s) are missing shards of "
                         f"the n{num_shards} sharding: "
                         f"{ {ds: m[:6] for ds, m in list(incomplete.items())[:4]} }"
                         f"{' ...' if len(incomplete) > 4 else ''}")
    destination = os.path.join(out_dir, "manifests", f"{name}.jsonl")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("wb", dir=os.path.dirname(destination),
                                      prefix=f".{name}.", suffix=".jsonl", delete=False)
    episodes = 0
    try:
        for ds in sorted(groups):
            for path in groups[ds]:
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 23), b""):
                        episodes += chunk.count(b"\n")
                        tmp.write(chunk)
        tmp.close()
        os.replace(tmp.name, destination)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    return {"combined": destination, "datasets": len(groups), "episodes": episodes,
            "num_shards": num_shards,
            "shards_seen": sorted({_part_shard(p) for ps in groups.values() for p in ps}),
            "bytes": os.path.getsize(destination)}


def write_report(path: str, payload: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", dir=directory, prefix=".report.", suffix=".json",
                                      delete=False, encoding="utf-8")
    try:
        json.dump(payload, tmp, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.write("\n")
        tmp.close()
        os.replace(tmp.name, path)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--latent-dir", default=DEFAULT_LATENT_DIR,
                    help="directory holding <Dataset>/<episode_id>/<view>/latent.npz")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--name", default="rynnvla_base", help="combined manifest basename")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset names; default all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split every dataset by source-JSON position; run one job per shard")
    ap.add_argument("--expect-shards", type=int, default=0,
                    help="combine: fail when a dataset has fewer parts than this")
    ap.add_argument("--workers", type=int, default=64,
                    help="measured optimum; the mount stops scaling past ~64")
    ap.add_argument("--limit", type=int, default=0, help="per-dataset cap on kept episodes, 0 = all")
    ap.add_argument("--max-caption-chars", type=int, default=512)
    ap.add_argument("--tail-bytes", type=int, default=1 << 16,
                    help="bytes read from the end of each npz to find its meta member")
    ap.add_argument("--expected-latent-dim", type=int, default=608,
                    help="reject episodes whose code_shape differs (0 disables)")
    ap.add_argument("--expected-pair-stride", type=int, default=0,
                    help="reject episodes labeled with a different pair_stride (0 disables)")
    ap.add_argument("--expected-representation", default=None,
                    help="reject episodes labeled with a different representation")
    ap.add_argument("--default-fps", type=float, default=30.0,
                    help="fps for a latent whose meta recorded none and whose dataset is not "
                         "in STATIC_FPS (build_index would apply its own default anyway)")
    ap.add_argument("--force", action="store_true", help="rebuild parts that already exist")
    ap.add_argument("--combine", action="store_true",
                    help="after rebuilding, concatenate every part of this sharding")
    ap.add_argument("--combine-only", action="store_true",
                    help="skip rebuilding; only concatenate the parts already on disk")
    ap.add_argument("--report", default=None,
                    help="default <out-dir>/rebuild_report.n<N>.shard<i>.json")
    args = ap.parse_args()

    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard {args.shard} out of range for --num-shards {args.num_shards}")
    args.latent_dir = os.path.abspath(args.latent_dir)
    report_path = args.report or os.path.join(
        args.out_dir, f"rebuild_report.n{args.num_shards}.shard{args.shard:04d}.json")

    if args.combine_only:
        result = combine_parts(args.out_dir, args.name, args.num_shards, args.expect_shards)
        if not result:
            raise SystemExit(f"no n{args.num_shards} parts under {args.out_dir}/manifests/parts "
                             "-- rebuild first")
        print(f"[combine] {json.dumps(result, sort_keys=True)}")
        return

    if not os.path.isdir(args.latent_dir):
        raise SystemExit(f"latent dir not found: {args.latent_dir}")
    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets else None
    files = discover_datasets(args.data_root, wanted)
    print(f"[rebuild] shard {args.shard}/{args.num_shards}: {len(files)} dataset JSON(s), "
          f"latents under {args.latent_dir}, workers={args.workers}", flush=True)

    opt = Options(args)
    reports, total_kept = [], 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for dataset, path in sorted(files.items()):
            report = rebuild_dataset(dataset, path, args.out_dir, args, opt, executor)
            reports.append(report)
            total_kept += report["kept"]
            print(f"[rebuild] {dataset}: kept={report['kept']:,}"
                  + ("" if report["reused"] else f" scanned={report['scanned']:,}")
                  + f" in {report['seconds']}s"
                  + (f" skips={report['skips']}" if report["skips"] else "")
                  + (" (reused part)" if report["reused"] else ""), flush=True)
            # Rewritten after every dataset: a full sweep runs for hours and this is the only
            # live view of what is done and why episodes were dropped.
            write_report(report_path, {
                "shard": args.shard, "num_shards": args.num_shards,
                "data_root": args.data_root, "latent_dir": args.latent_dir,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
                "kept_total": total_kept, "datasets": reports,
            })

    skips = Counter()
    for report in reports:
        skips.update(report.get("skips") or {})
    print(f"[rebuild] kept {total_kept:,} episodes; skips {dict(skips.most_common(12))}", flush=True)
    if any(key.startswith("unmapped_camera:") for key in skips):
        print("[rebuild] FATAL: unmapped camera names -- extend constants.VIEW_NAME_TO_ROLE; "
              "build_index raises on them, so no index could be built.", file=sys.stderr, flush=True)
        sys.exit(2)

    if args.combine:
        result = combine_parts(args.out_dir, args.name, args.num_shards, args.expect_shards)
        if result:
            print(f"[combine] {json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
