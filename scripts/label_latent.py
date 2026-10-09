#!/usr/bin/env python3
"""Label local videos with RynnLAM latent actions -> per-(episode, view) latent.npz.

Minimal single-node CLI. It decodes a local video with `rynnlam.video.FrameReader`, encodes
each ``(t, t+gap)`` frame pair (advancing by ``pair_stride``) with a frozen `RynnLAMEncoder`,
and writes the npz that the Stage-1 index builders consume:

    video --(this script)--> latent.npz
        -> scripts/rebuild_latent_manifest.py     (manifest jsonl)
        -> scripts/build_latent_pretrain_index.py (Stage-1 training index npz)
        -> scripts/build_latent_stats.py          (normalization stats json)
        -> rynnvla.api.train                       (Stage-1 latent-action pretraining)

The byte layout and ``meta`` match the "Latent-action protocol" section of README.md
(``schema_version 2``, ``gap 4``, ``pair_stride 4``, ``latent_action`` float16 ``[N, code]``,
``pair_indices`` int32 ``[N, 2]``, ``meta`` a JSON string). The default representation is
``ktoken_zcam`` (608 dims for the b512 recipe: 8*64 k-tokens + 64 z + 32 camera_pose), which
is what Stage-1 consumes.

This needs a trained RynnLAM motion checkpoint (``--checkpoint``); producing one is
`scripts/train_rynnlam.py`. Run from the repo root: relative ``video_path`` values resolve
against the current working directory, as they do everywhere else in this repo. Examples:

    # one video -> <out>/latents/videos/<stem>/<view>/latent.npz
    python scripts/label_latent.py --checkpoint rynnlam_motion.pt \
        --video demo.mp4 --output-dir out

    # a RynnVLA-Base episode JSON (list of {dataset, episode_id, views:{view:{video_path}}})
    # -> <out>/latents/<dataset>/<episode_id>/<view>/latent.npz
    python scripts/label_latent.py --checkpoint rynnlam_motion.pt \
        --metadata data/sample/source/SampleToy.json --output-dir out
"""
import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rynnlam.inference import RynnLAMEncoder, file_sha256, implementation_sha256
from rynnlam.video import BUCKETS, FrameReader, pair_batches, resolve_source


def safe_name(value, *, nested=False):
    """Reject path-traversal in dataset/episode/view identifiers used to build output paths."""
    value = str(value)
    path = PurePosixPath(value)
    if (
        not value
        or value in {".", ".."}
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
    ):
        raise ValueError(f"Invalid output identifier: {value!r}")
    if not nested and "/" in value:
        raise ValueError(f"Expected one path component: {value!r}")
    return value


def label_video(encoder, video, protocol, *, view, batch_size):
    """Encode every (t, t+gap) pair of one video; return (latent, pair_indices, meta)."""
    reader = FrameReader(video, view)
    latents, pair_indices = [], []
    shape = bucket = native_hw = None
    for images, pairs, bkt, nhw in pair_batches(
        reader,
        gap=protocol["gap"],
        batch_size=batch_size,
        bucket=protocol["bucket"],
        start=protocol["start_frame"],
        end=protocol["end_frame"],
        pair_stride=protocol["pair_stride"],
    ):
        tokens = encoder(images, representation=protocol["representation"]).cpu().numpy()
        current = list(tokens.shape[1:])
        if shape is not None and current != shape:
            raise ValueError("Token shape changed within the video")
        shape = current
        encoded = tokens.reshape(len(tokens), -1).astype(np.float16)
        if not np.isfinite(encoded).all():
            raise ValueError("Non-finite values after float16 conversion")
        latents.append(encoded)
        pair_indices.append(pairs)
        bucket, native_hw = bkt, nhw
    if not latents:
        raise ValueError(f"No frame pairs encoded for {video}")
    latent = np.concatenate(latents).astype(np.float16)
    pairs = np.concatenate(pair_indices).astype(np.int32)
    count = len(latent)
    num_frames = reader.num_frames
    meta = {
        "protocol": protocol,
        "view": view,
        "num_latents": count,
        "num_frames": num_frames,
        "unpaired_tail_frames": num_frames
        - ((count - 1) * protocol["pair_stride"] + protocol["gap"] + 1),
        "code_shape": shape,
        "bucket": bucket,
        "target_hw": list(BUCKETS[bucket]),
        "source_hw": list(native_hw),
        "fps": reader.fps,
    }
    return latent, pairs, meta


def write_npz(destination, latent, pairs, meta):
    """np.savez seeks on write, which FUSE mounts reject; build in memory then write bytes.

    The bytes land in a ``.partial`` sibling and are published with ``os.replace``, so this
    writer can never leave a half-written file at the final path -- not for its own skip check
    on a rerun, and not for a ``rebuild_latent_manifest`` sweep running while labeling is still
    in progress. ``npz_is_complete`` is the second layer, covering files truncated by something
    other than this function. Not imported from ``stream_extract_tar.atomic_write`` because that
    module pulls in cv2 and torch at import time, which the labeler does not otherwise need.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez(buffer, latent_action=latent, pair_indices=pairs, meta=json.dumps(meta))
    payload = buffer.getvalue()
    pending = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        pending.write_bytes(payload)
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def npz_is_complete(path):
    """Cheap truncation probe for an existing ``latent.npz``.

    Constructing a ``ZipFile`` reads only the End-Of-Central-Directory record at the tail, so
    this costs one small read rather than the whole payload -- the same reason
    ``rebuild_latent_manifest.read_npz_meta`` reads a tail window. A truncated file raises
    ``BadZipFile``, which subclasses ``Exception`` directly and is therefore caught by neither
    ``OSError`` nor ``ValueError``; naming it explicitly is the whole point.

    The member is ``latent_action.npy``, not ``latent_action``: ``np.savez`` appends ``.npy`` to
    every key, so testing for the bare array name would report every healthy file as corrupt and
    relabel the whole corpus on each run.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            return "latent_action.npy" in archive.namelist()
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def build_items(args):
    if args.metadata:
        with args.metadata.open(encoding="utf-8") as handle:
            items = json.load(handle)
        if not isinstance(items, list):
            raise SystemExit("--metadata must contain a JSON list of episode entries")
        # Relative video_path values resolve against the current working directory, matching
        # the Stage-1 consumer (latent_pretrain.py) and every committed config/manifest in
        # this repo -- so run from the repo root, as documented in README.md.
        return items, Path.cwd()
    item = {
        "dataset": "videos",
        "episode_id": args.video.stem,
        "views": {args.view: {"video_path": str(args.video.resolve())}},
    }
    return [item], Path.cwd()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="One RGB video -> a single latent.npz")
    source.add_argument(
        "--metadata", type=Path,
        help="JSON list of {dataset, episode_id, views:{view:{video_path,...}}} entries",
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Trained RynnLAM motion checkpoint (from scripts/train_rynnlam.py)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--representation",
                        choices=["ktoken", "z", "zcam", "hints_pool", "ktoken_zcam"],
                        default="ktoken_zcam")
    parser.add_argument("--view", default="head", help="View name used with --video")
    parser.add_argument("--views", nargs="+", help="Only label these view names from --metadata")
    parser.add_argument("--gap", type=int, default=4, help="RGB frame interval within each pair")
    parser.add_argument("--pair-stride", "--stride", dest="pair_stride", type=int, default=4,
                        help="Step between pair start frames; default 4 gives (0,4),(4,8),...")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bucket", choices=["auto", *BUCKETS], default="auto")
    parser.add_argument("--normalization", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override fps in meta when the container reports none")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-label even if the destination latent.npz already exists")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow Python object deserialization; only for a trusted checkpoint")
    args = parser.parse_args(argv)

    if args.gap < 1 or args.pair_stride < 1 or args.batch_size < 1:
        parser.error("--gap, --pair-stride and --batch-size must be positive")

    items, base = build_items(args)
    checkpoint_hash = file_sha256(args.checkpoint)
    implementation_hash = implementation_sha256()
    labeler_hash = file_sha256(Path(__file__).resolve())
    encoder = RynnLAMEncoder(
        args.checkpoint, device=args.device,
        normalize=args.normalization == "imagenet", precision=args.precision,
        trust_checkpoint=args.trust_checkpoint, checkpoint_sha256=checkpoint_hash,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Recorded rather than left to be inferred from the bytes, because it cannot be: bf16 matmul
    # is hardware-dependent, so the same checkpoint at the same --precision produces different
    # bits on different GPUs. The logical device alone ("cuda:0") does not say which silicon did
    # the work, hence the name beside it. This is provenance, not geometry -- the bytes and their
    # meaning are unchanged, so schema_version stays at 2.
    device_name = (torch.cuda.get_device_name()
                   if encoder.device.type == "cuda" else "cpu")

    labeled = skipped = 0
    for item in items:
        dataset = safe_name(item["dataset"])
        episode = safe_name(item["episode_id"], nested=True)
        for view, source_view in item["views"].items():
            safe_name(view)
            if args.views and view not in args.views:
                continue
            destination = args.output_dir / "latents" / dataset / episode / view / "latent.npz"
            if destination.exists() and not args.overwrite:
                if npz_is_complete(destination):
                    print(f"[skip] {dataset}/{episode}/{view}: exists (use --overwrite)")
                    skipped += 1
                    continue
                # Existence alone used to be the entire test, so a file truncated by an
                # interrupted run was skipped forever and only surfaced much later as an
                # `unreadable_latent` drop in rebuild_latent_manifest -- a silent hole in the
                # corpus. Relabel it instead of trusting it.
                print(f"[relabel] {dataset}/{episode}/{view}: existing latent.npz is "
                      f"truncated or unreadable")
            video, source_size, source_mtime_ns = resolve_source(source_view["video_path"], base)
            start = int(source_view.get("start_frame", item.get("start_frame", 0)) or 0)
            end = source_view.get("end_frame", item.get("end_frame"))
            protocol = {
                "schema_version": 2,
                "checkpoint_sha256": checkpoint_hash,
                "implementation_sha256": implementation_hash,
                "labeler_sha256": labeler_hash,
                "source": str(video),
                "source_size": source_size,
                "source_mtime_ns": source_mtime_ns,
                "gap": args.gap,
                "pair_stride": args.pair_stride,
                "start_frame": start,
                "end_frame": int(end) if end is not None else None,
                "bucket": args.bucket,
                "normalization": args.normalization,
                "representation": args.representation,
                "precision": args.precision,
                "device": str(encoder.device),
                "device_name": device_name,
            }
            latent, pairs, meta = label_video(
                encoder, video, protocol, view=view, batch_size=args.batch_size
            )
            if args.fps is not None and not meta.get("fps"):
                meta["fps"] = args.fps
            write_npz(destination, latent, pairs, meta)
            labeled += 1
            print(f"{dataset}/{episode}/{view}: {meta['num_latents']} latents, "
                  f"shape={meta['code_shape']} -> {destination}", flush=True)

    latents = args.output_dir / "latents"
    build = args.output_dir / "build"
    manifest = build / "manifests" / "corpus.jsonl"
    # --data-root is the directory holding the <Dataset>.json episode files, which is the
    # parent of --metadata when one was given.
    data_root = args.metadata.parent if args.metadata else Path("<dir holding Dataset.json>")
    print(f"\n[label] wrote {labeled} latent.npz ({skipped} skipped) under {latents}")
    print("[label] next -- build the Stage-1 index from these latents:")
    print(f"    python scripts/rebuild_latent_manifest.py --data-root {data_root} \\")
    print(f"        --latent-dir {latents} --out-dir {build} --name corpus --combine")
    print(f"    python scripts/build_latent_pretrain_index.py --manifest {manifest} \\")
    print(f"        --out {build / 'index_c6.npz'} --latent-chunk 6")
    print(f"    python scripts/build_latent_stats.py --manifest {manifest} \\")
    print(f"        --out {build / 'latent_stats.json'} --latent-dim 608")


if __name__ == "__main__":
    main()
