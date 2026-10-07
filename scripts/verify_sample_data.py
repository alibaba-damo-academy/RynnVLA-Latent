#!/usr/bin/env python3
"""Verify the synthetic Stage-1 sample loads through the real production data pipeline.

This is the no-backbone, no-GPU check: it builds `LatentPretrainDataset` via the
public `rynnvla.datasets.build_dataset` entry with the generated mixture and reads real
samples, asserting that decoded images and finite 608-dim latent targets come back. It
proves the *data* half of Stage 1 runs end to end on a fresh clone.

It does NOT build the RynnVLA model (that needs a RynnBrain backbone). For a full forward
/ optimizer smoke, see the training command printed by scripts/make_sample_data.py.

Usage:
    python scripts/verify_sample_data.py                       # data/sample/sample_mixture.json
    python scripts/verify_sample_data.py --mixture /path/to/sample_mixture.json --num-samples 5
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force CPU: the data pipeline must not need a GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from rynnvla.datasets import build_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mixture", default=str(REPO_ROOT / "data" / "sample" / "sample_mixture.json"))
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--latent-dim", type=int, default=608)
    ap.add_argument("--latent-chunk", type=int, default=6)
    args = ap.parse_args()

    mixture_path = Path(args.mixture).resolve()
    if not mixture_path.is_file():
        raise SystemExit(
            f"mixture not found: {mixture_path}\n"
            f"  Generate it first:  python scripts/make_sample_data.py"
        )
    mixture = json.loads(mixture_path.read_text())

    # The committed sample stores repo-root-relative paths (see make_sample_data.py) and the
    # dataset resolves them against the process cwd, so anchor here: this check then passes
    # from any directory rather than only from the repository root.
    os.chdir(REPO_ROOT)

    # Processing knobs mirror rynnvla/configs/stage1_smoke_tiny.json. processor stays None, so
    # __getitem__ returns the raw sample dict (images / latent_targets / slot_mask / text).
    # eef_rotation_repr=None: latent pretraining carries no robot-action rotation, and
    # BaseVLADataset asserts it is either None or a RotationRepresentation enum.
    ds_args = SimpleNamespace(
        model_max_length=4096, mm_max_length=64, fps=2, max_frames=64,
        action_chunk_size=30, use_delta_action=False, eef_rotation_repr=None,
        action_only=False, target_fps=None, num_view_slots=6,
        latent_action_dim=args.latent_dim, use_visual_augmentation=False,
        chunk_overlap_ratio=0.0, emit_teacher_images=False, data_mixture=mixture,
    )
    dataset = build_dataset(ds_args)
    n = len(dataset)
    print(f"[verify] dataset built: {n} samples")
    if n == 0:
        raise SystemExit("[verify] FAIL: dataset is empty")

    k = min(args.num_samples, n)
    for i in range(k):
        sample = dataset[i]
        for key in ("images", "latent_targets", "slot_mask", "text"):
            if key not in sample:
                raise SystemExit(f"[verify] FAIL: sample {i} missing {key!r}; got {sorted(sample)}")
        latents = sample["latent_targets"]
        latents = latents if isinstance(latents, torch.Tensor) else torch.as_tensor(latents)
        if not torch.isfinite(latents).all():
            raise SystemExit(f"[verify] FAIL: sample {i} latent_targets has non-finite values")
        if latents.shape[-1] != args.latent_dim:
            raise SystemExit(f"[verify] FAIL: sample {i} latent dim {latents.shape[-1]} != {args.latent_dim}")
        images = sample["images"]
        if not images:
            raise SystemExit(f"[verify] FAIL: sample {i} has no images")
        view, frame = next(iter(images.items()))
        frame = frame if isinstance(frame, torch.Tensor) else torch.as_tensor(frame)
        if not torch.isfinite(frame.float()).all():
            raise SystemExit(f"[verify] FAIL: sample {i} view {view} frame has non-finite pixels")
        print(f"[verify] sample {i}: latent_targets {tuple(latents.shape)} (finite), "
              f"views={sorted(images)} frame[{view}]={tuple(frame.shape)}, "
              f"slot_mask={sample['slot_mask'].sum().item()} active, text={sample['text'][:40]!r}")

    print(f"[verify] OK — read {k}/{n} samples through LatentPretrainDataset; "
          f"images decoded, {args.latent_dim}-dim latent targets finite, no GPU.")


if __name__ == "__main__":
    main()
