#!/usr/bin/env python3
"""Generate a tiny synthetic RynnLAM training corpus under data/sample_lam/.

RynnLAM trains on *preprocessed per-scene safetensors*, which is a different format from the
Stage-1 latent npz under data/sample/ -- see the "Latent-action protocol" and "Latent-action
model (RynnLAM)" sections of README.md. This script materializes the smallest corpus that
exercises the real training path, so a fresh clone can run RynnLAM without downloading or
preprocessing anything:

    data/sample_lam/
    ├── manifest_0000.json                       # scene index; `file` is manifest-relative
    └── scenes/SampleToy__ep{i}__head.safetensors

Each safetensors holds one scene, tensors keyed by "<file stem>__<field>":

    __rgb         uint8   [T, H, W, 3]
    __depth       float16 [T, H, W]        must be > 0 where valid (the flow mask is derived from it)
    __flow        float16 [T-1, H, W, 3]   precomputed 3D flow; the last-dim==3 branch is taken
    __extrinsics  float32 [T, 4, 4]
    __intrinsics  float32 [T, 3, 3]
    __mask        uint8   [T, H, W]

The frames are a translating gradient and the flow/depth are smooth, so every loss branch sees
non-degenerate input. The values are SYNTHETIC and carry no real signal: they prove the pipeline
and model *run*, not that a latent action model learns.

The encoder is randomly initialized by default because the DA3-Large weights are not
redistributed; pass --encoder-checkpoint to train against the real backbone.

Usage:
    python scripts/make_sample_lam_data.py                    # 3 scenes -> data/sample_lam/
    python scripts/make_sample_lam_data.py --scenes 1 --out data/sample_lam_custom

Then train (2 optimizer steps, single GPU):
    python scripts/train_rynnlam.py --config rynnlam/configs/smoke.yaml
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET = "SampleToy"
VIEW = "head"
BUCKET = "square"
PATCH_SIZE = 14  # DA3-Large patch size; --hw must be a multiple of it
# Smallest frame count that still admits one sampled pair at a recipe's min_frame_stride.
MIN_FRAMES = 6


def _rel(path: Path) -> str:
    """Repo-root-relative when inside the repo, else absolute (keeps the sample portable)."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def make_scene(num_frames: int, hw: int, seed: int):
    """Synthetic (rgb, depth, flow, extrinsics, intrinsics, mask) for one scene."""
    rng = np.random.default_rng(seed)
    t, h, w = num_frames, hw, hw

    # Translating gradient: real inter-frame motion, so the flow target is not all zeros.
    base = np.linspace(0, 255, w, dtype=np.float32)
    grid = base[None, :] * 0.6 + base[:, None] * 0.4
    rgb = np.empty((t, h, w, 3), np.uint8)
    for i in range(t):
        shift = (i * 4) % w
        frame = np.roll(grid, shift, axis=1)
        frame = np.stack([frame, np.roll(frame, w // 3), np.roll(frame, 2 * w // 3)], -1)
        rgb[i] = np.clip(frame + rng.normal(0, 4, (h, w, 3)), 0, 255).astype(np.uint8)

    # Smooth positive depth in [0.8, 2.2] m; the loader derives the flow mask from depth > 0.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = (1.5 + 0.5 * np.sin(xx / max(w - 1, 1) * np.pi)
             + 0.2 * np.cos(yy / max(h - 1, 1) * np.pi)).astype(np.float32)
    depth = np.repeat(depth[None], t, axis=0).astype(np.float16)

    # Small 3D flow per step, dominated by the horizontal translation used above.
    flow = np.zeros((t - 1, h, w, 3), np.float32)
    flow[..., 0] = 0.02
    flow[..., 2] = rng.normal(0, 0.002, (t - 1, h, w)).astype(np.float32)
    flow = flow.astype(np.float16)

    extrinsics = np.tile(np.eye(4, dtype=np.float32), (t, 1, 1))
    extrinsics[:, 0, 3] = np.linspace(0, 0.05, t).astype(np.float32)

    focal = float(max(h, w))
    k = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], np.float32)
    intrinsics = np.repeat(k[None], t, axis=0).copy()

    mask = np.full((t, h, w), 255, np.uint8)
    return rgb, depth, flow, extrinsics, intrinsics, mask


def write_scene(path: Path, stem: str, arrays) -> None:
    """Write one scene safetensors.

    Serialized to a local temp file then moved: safetensors' writer is rejected with EACCES by
    some network/FUSE mounts even when the destination directory is world-writable.
    """
    from safetensors.numpy import save_file

    rgb, depth, flow, extrinsics, intrinsics, mask = arrays
    tensors = {
        f"{stem}__rgb": rgb,
        f"{stem}__depth": depth,
        f"{stem}__flow": flow,
        f"{stem}__extrinsics": extrinsics,
        f"{stem}__intrinsics": intrinsics,
        f"{stem}__mask": mask,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lam_scene_") as tmp:
        staging = Path(tmp) / path.name
        save_file(tensors, str(staging))
        shutil.move(str(staging), str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "sample_lam"),
                    help="output directory (created fresh; must not exist)")
    ap.add_argument("--scenes", type=int, default=3, help="number of synthetic scenes (1-5)")
    ap.add_argument("--num-frames", type=int, default=12)
    ap.add_argument("--hw", type=int, default=56,
                    help=f"square frame size; must be a multiple of {PATCH_SIZE}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not 1 <= args.scenes <= 5:
        raise SystemExit("--scenes must be in 1..5")
    if args.num_frames < MIN_FRAMES:
        raise SystemExit(f"--num-frames must be >= {MIN_FRAMES}")
    if args.hw % PATCH_SIZE:
        raise SystemExit(f"--hw must be a multiple of {PATCH_SIZE} (the DA3-Large patch size)")

    out = Path(args.out).resolve()
    if os.path.lexists(out):
        raise SystemExit(f"--out already exists: {out}\n  Remove it or pass a new --out.")
    scenes_dir = out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(args.scenes):
        scene_id = f"{DATASET}/ep{i}/{VIEW}"
        stem = scene_id.replace("/", "__")
        # Resolved against this manifest's own directory by LAMEpicDataset, so the corpus stays
        # valid wherever the repo is checked out.
        file_rel = f"scenes/{stem}.safetensors"
        arrays = make_scene(args.num_frames, args.hw, seed=args.seed + i)
        write_scene(scenes_dir / f"{stem}.safetensors", stem, arrays)
        entries.append({
            "scene_id": scene_id,
            "file": file_rel,
            "num_frames": args.num_frames,
            "num_flows": args.num_frames - 1,
            "height": args.hw,
            "width": args.hw,
            "bucket": BUCKET,
            "depth_stride": 1,
        })
        size = (scenes_dir / f"{stem}.safetensors").stat().st_size
        print(f"[lam-sample] {scene_id}: {args.num_frames}x{args.hw}x{args.hw} "
              f"({size / 1024:.0f} KB)")

    manifest = out / "manifest_0000.json"
    manifest.write_text(json.dumps(entries, indent=2) + "\n")

    print(f"\n[lam-sample] wrote {args.scenes} scene(s) under {_rel(out)}/")
    print(f"[lam-sample] manifest: {_rel(manifest)}")
    print("[lam-sample] next -- train RynnLAM for 2 steps on this corpus:")
    print("    python scripts/train_rynnlam.py --config rynnlam/configs/smoke.yaml")


if __name__ == "__main__":
    main()
