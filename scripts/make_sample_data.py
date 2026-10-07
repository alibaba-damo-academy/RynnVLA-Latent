#!/usr/bin/env python3
"""Generate a tiny, self-contained synthetic Stage-1 sample under data/sample/.

Goal: let someone who cloned this repo exercise the real Stage-1 latent-action data
pipeline -- and, if they supply a RynnBrain backbone, a short training smoke run -- with
NO corpus download and NO OSS credentials.

The data is SYNTHETIC: solid/gradient frames and random 608-dim latents that match the
`ktoken_zcam` schema (the "Latent-action protocol" section of README.md) byte-for-byte in
layout, but carry no real signal. It proves the pipeline and model *run*; it is not meant to
train a useful policy.

The generated tree under data/sample/ IS committed, so a fresh clone can run Stage 1 with no
generator step. Every path it contains is therefore written repo-root-relative, which keeps
the sample valid no matter where the repo is checked out; run the consumers (verify script,
torchrun) from the repository root so those relative paths resolve. Pass --out outside the
repo to get absolute paths instead. It reuses the repository's own builders as the single
source of truth:

    source JSON + latent.npz + videos
      -> scripts/rebuild_latent_manifest.py   (manifest JSONL)
      -> scripts/build_latent_pretrain_index.py (index npz, version 4)
      -> scripts/build_latent_stats.py          (latent_stats.json)
      -> sample_mixture.json                    (LatentPretrainDataset mixture)

Usage:
    python scripts/make_sample_data.py                 # 3 episodes -> data/sample/
    python scripts/make_sample_data.py --episodes 1 --out data/sample_custom

Then verify the data pipeline (CPU, no backbone, no GPU):
    python scripts/verify_sample_data.py --mixture data/sample/sample_mixture.json
"""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

DATASET = "SampleToy"
VIEW = "head"
LATENT_DIM = 608          # ktoken_zcam: 512 k_tokens + 64 z + 32 camera_pose
K_DIM, Z_DIM, CAM_DIM = 512, 64, 32
REPRESENTATION = "ktoken_zcam"
SCHEMA_VERSION = 2
# 64x64 is square, so this is the bucket scripts/label_latent.py would resolve it to.
BUCKET = "square"
# Stands in for the three producer-identity hashes. Nothing encoded these latents, so there
# is no real checkpoint / implementation / labeler to fingerprint.
SYNTHETIC = "synthetic-sample"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# rynnlam.video is torch-free, so the bucket table can be shared with the real labeler
# instead of duplicated here (and drifting).
from rynnlam.video import BUCKETS  # noqa: E402


def _rel(path: Path) -> str:
    """Repo-root-relative when the sample lives inside the repo, else absolute.

    The committed data/sample must stay valid after any clone, so its paths cannot carry
    this machine's checkout prefix. An --out outside the repo has no relative form.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def write_video(path: Path, num_frames: int, hw: int, seed: int) -> None:
    """Encode a tiny deterministic mp4 that PyAV (the loader's decoder) can read.

    Encoded to a local temp file then moved: the MP4 muxer seeks on write, which FUSE
    mounts reject with OSError(95) -- the same reason build_latent_pretrain_index.py
    builds its npz in a TemporaryDirectory before copying it into place.
    """
    import cv2

    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, hw, dtype=np.float32)
    grid = (base[None, :] * 0.6 + base[:, None] * 0.4)
    tmp_dir = tempfile.mkdtemp(prefix="sample_vid_")
    tmp_path = os.path.join(tmp_dir, "clip.mp4")
    try:
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (hw, hw))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {tmp_path}")
        try:
            for t in range(num_frames):
                shift = (t * 3) % hw
                frame = np.roll(grid, shift, axis=1)
                frame = np.stack([frame, np.roll(frame, hw // 3), np.roll(frame, 2 * hw // 3)], -1)
                frame = np.clip(frame + rng.normal(0, 4, (hw, hw, 3)), 0, 255).astype(np.uint8)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_path, str(path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def latent_meta(num_latents: int, num_frames: int, hw: int, fps: float,
                gap: int, pair_stride: int, source: str,
                source_size: int, source_mtime_ns: int) -> dict:
    """The `meta` member, key-for-key what scripts/label_latent.py writes.

    Keeping the two in sync is the point: the bundled sample is the format example a
    third party re-labeling a corpus will read, so it must show the real schema rather
    than a subset of it. The three ``*_sha256`` fields read ``synthetic-sample`` because
    no encoder produced these latents -- there is no checkpoint, implementation or labeler
    to fingerprint. ``protocol.bucket`` is the *requested* bucket ("auto", the labeler's
    default) and the top-level ``bucket`` is the *resolved* one, exactly as in real output.
    ``protocol.device`` and ``protocol.device_name`` read ``cpu`` because these arrays come
    from numpy rather than an encoder forward pass, so there is no accelerator to name.
    """
    unpaired_tail = num_frames - ((num_latents - 1) * pair_stride + gap + 1)
    return {
        "protocol": {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_sha256": SYNTHETIC,
            "implementation_sha256": SYNTHETIC,
            "labeler_sha256": SYNTHETIC,
            "source": source,
            "source_size": source_size,
            "source_mtime_ns": source_mtime_ns,
            "gap": gap,
            "pair_stride": pair_stride,
            "start_frame": 0,
            "end_frame": None,
            "bucket": "auto",
            "normalization": "imagenet",
            "representation": REPRESENTATION,
            "precision": "bf16",
            "device": "cpu",
            "device_name": "cpu",
        },
        "view": VIEW,
        "num_latents": num_latents,
        "num_frames": num_frames,
        "unpaired_tail_frames": unpaired_tail,
        "code_shape": [LATENT_DIM],
        "bucket": BUCKET,
        "target_hw": list(BUCKETS[BUCKET]),
        "source_hw": [hw, hw],
        "fps": fps,
    }


def write_latent(path: Path, num_latents: int, num_frames: int, hw: int, fps: float,
                 gap: int, pair_stride: int, video_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.0, size=(num_latents, LATENT_DIM)).astype(np.float16)
    starts = np.arange(num_latents) * pair_stride
    pairs = np.stack([starts, starts + gap], axis=1).astype(np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    stat = video_path.stat()
    # Build the npz in memory then write bytes sequentially: np.savez seeks on write,
    # which FUSE mounts reject (OSError 95).
    buf = io.BytesIO()
    np.savez(buf, latent_action=latent, pair_indices=pairs,
             meta=json.dumps(latent_meta(num_latents, num_frames, hw, fps, gap, pair_stride,
                                         _rel(video_path), stat.st_size, stat.st_mtime_ns)))
    path.write_bytes(buf.getvalue())


def run_builder(script: str, args: list) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), *args]
    print(f"[sample] $ {script} {' '.join(args)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"[sample] builder failed: {script} (exit {result.returncode})")
    tail = "\n".join(result.stdout.strip().splitlines()[-3:])
    if tail:
        print(f"[sample]   {tail}")


def _to_rel(text: str) -> str:
    """Absolute path under the repo -> repo-root-relative; anything else passes through."""
    if not text.startswith("/"):
        return text
    try:
        return str(Path(text).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return text


def _relocate(out: Path, index: Path, stats: Path, manifest: Path, build: Path) -> None:
    """Rewrite the builders' absolute paths to repo-root-relative and drop intermediates.

    ``rebuild_latent_manifest.py`` abspaths ``--latent-dir`` and both stats/index builders
    record the manifest they read, so the artifacts they emit always carry this machine's
    checkout prefix. The committed sample has to survive a clone to a different prefix.
    """
    from rynnvla.datasets.vla_datasets.latent_pretrain import _StringColumn, _unpack_strings

    def pack(values):
        column = _StringColumn()
        for text in values:
            column.append(text)
        return column.arrays()

    with np.load(index) as z:
        arrays = {key: z[key] for key in z.files}
        videos = [_to_rel(v) for v in _unpack_strings(z["video_blob"], z["video_pos"])]
        latents = [_to_rel(v) for v in _unpack_strings(z["latent_blob"], z["latent_pos"])]
        meta = json.loads(str(z["meta"][0]))

    meta["manifest"] = _to_rel(str(meta.get("manifest") or manifest))
    arrays["video_blob"], arrays["video_pos"] = pack(videos)
    arrays["latent_blob"], arrays["latent_pos"] = pack(latents)
    arrays["meta"] = np.array([json.dumps(meta)])
    # BytesIO then write_bytes: np.savez seeks on write, which FUSE mounts reject.
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    index.write_bytes(buf.getvalue())

    stats_obj = json.loads(stats.read_text())
    if "manifest" in stats_obj:
        stats_obj["manifest"] = _to_rel(str(stats_obj["manifest"]))
    stats.write_text(json.dumps(stats_obj))

    # The combined manifest stays as the index's provenance (meta.manifest points at it);
    # the per-shard parts and the rebuild report are build intermediates nothing reads.
    lines = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        for view in entry.get("views") or []:
            for key in ("video_path", "latent_path"):
                if isinstance(view.get(key), str):
                    view[key] = _to_rel(view[key])
        lines.append(json.dumps(entry, ensure_ascii=False))
    manifest.write_text("\n".join(lines) + "\n")

    shutil.rmtree(build / "manifests" / "parts", ignore_errors=True)
    for report in build.glob("rebuild_report*.json"):
        report.unlink()
    print(f"[sample] relocated paths under {_rel(out)}/ to repo-root-relative")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "sample"),
                    help="output directory (created fresh; must not exist)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--num-latents", type=int, default=24, help="latent rows per episode")
    ap.add_argument("--hw", type=int, default=64, help="square frame size of the synthetic video")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--gap", type=int, default=4)
    ap.add_argument("--pair-stride", type=int, default=4)
    ap.add_argument("--latent-chunk", type=int, default=6)
    ap.add_argument("--latent-step-seconds", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")

    out = Path(args.out).resolve()
    if os.path.lexists(out):
        raise SystemExit(f"--out already exists: {out}\n  Remove it or pass a new --out.")

    videos = out / "videos"
    latents = out / "latents"
    source = out / "source"
    build = out / "build"
    for d in (videos, latents, source, build):
        d.mkdir(parents=True, exist_ok=True)

    num_frames = args.pair_stride * (args.num_latents - 1) + args.gap + 1 + 3
    episodes = []
    for i in range(args.episodes):
        ep = f"ep{i}"
        video_path = videos / f"{DATASET}_{ep}_{VIEW}.mp4"
        write_video(video_path, num_frames, args.hw, seed=args.seed + i)
        latent_path = latents / DATASET / ep / VIEW / "latent.npz"
        write_latent(latent_path, args.num_latents, num_frames, args.hw, args.fps,
                     args.gap, args.pair_stride, video_path, seed=args.seed + 1000 + i)
        episodes.append({
            "dataset": DATASET,
            "episode_id": ep,
            "caption": [{"start_time": 0.0,
                         "description": f"synthetic smoke sample episode {i}"}],
            "views": {VIEW: {"video_path": _rel(video_path),
                             "has_flow": False, "has_depth": False}},
        })
        print(f"[sample] {ep}: {num_frames}-frame {args.hw}x{args.hw} mp4 + "
              f"{args.num_latents}x{LATENT_DIM} latent")

    (source / f"{DATASET}.json").write_text(json.dumps(episodes, indent=2) + "\n")

    # 1) manifest from the synthetic source + latents (reuse the production builder).
    run_builder("rebuild_latent_manifest.py", [
        "--data-root", str(source), "--latent-dir", str(latents), "--out-dir", str(build),
        "--name", "sample", "--combine", "--workers", "1",
        "--expected-latent-dim", str(LATENT_DIM),
        "--expected-pair-stride", str(args.pair_stride),
        "--expected-representation", REPRESENTATION,
        "--default-fps", str(args.fps),
    ])
    manifest = build / "manifests" / "sample.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"[sample] expected combined manifest not found: {manifest}")

    # 2) index npz (version 4) + 3) latent_stats.json, both from the combined manifest.
    index = build / "index_c6.npz"
    stats = build / "latent_stats.json"
    run_builder("build_latent_pretrain_index.py", [
        "--manifest", str(manifest), "--out", str(index),
        "--latent-chunk", str(args.latent_chunk),
        "--latent-step-seconds", str(args.latent_step_seconds),
        "--default-fps", str(args.fps),
    ])
    run_builder("build_latent_stats.py", [
        "--manifest", str(manifest), "--out", str(stats), "--latent-dim", str(LATENT_DIM),
    ])

    # 4) strip this machine's checkout prefix out of the committed artifacts.
    _relocate(out, index, stats, manifest, build)

    # 5) a ready-to-use production mixture. Paths are repo-root-relative, so run the
    #    consumers from the repository root.
    mixture = [{
        "data_type": "LatentPretrainDataset",
        "data_path": _rel(index),
        "latent_stats_path": _rel(stats),
        "latent_chunk": args.latent_chunk,
        "latent_step_seconds": args.latent_step_seconds,
        "dataset_weights": {DATASET: 1.0},
    }]
    mixture_path = out / "sample_mixture.json"
    mixture_path.write_text(json.dumps(mixture, indent=2) + "\n")

    print(f"\n[sample] wrote {args.episodes} synthetic episode(s) under {_rel(out)}/")
    print(f"[sample] mixture: {_rel(mixture_path)}")
    print("[sample] next -- verify the data pipeline (CPU, no backbone), from the repo root:")
    print(f"    python scripts/verify_sample_data.py --mixture {_rel(mixture_path)}")
    print("[sample] next -- short training smoke with NO multi-GB backbone:")
    print("    python scripts/make_smoke_tiny_model.py --src <your local RynnBrain-2B or Qwen3-VL dir>")
    print("    torchrun --standalone --nproc_per_node=1 -m rynnvla.api.train \\")
    print("        --config rynnvla/configs/stage1_smoke_tiny.json \\")
    print("        --model_path data/smoke_tiny_model \\")
    print(f"        --data_mixture {_rel(mixture_path)} \\")
    print("        --output_dir runs/smoke_vla")


if __name__ == "__main__":
    main()
