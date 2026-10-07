#!/usr/bin/env python3
"""Build a production latent index from an explicit manifest (not a smoke subset).

This delegates to the dataset's build_index; it does not duplicate the builder.
The destination must be new and outside the source manifest directory.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--latent-chunk", type=int, default=6)
    parser.add_argument("--latent-step-seconds", type=float, default=0.25)
    parser.add_argument("--default-fps", type=float, default=30.0)
    args = parser.parse_args(argv)
    for path in (args.manifest, args.out):
        if not path.is_absolute() or ".." in path.parts:
            parser.error("--manifest and --out must be absolute paths without '..'")
    if not args.manifest.is_file():
        parser.error(f"Manifest does not exist: {args.manifest}")
    if args.out.suffix != ".npz":
        parser.error("--out must end in .npz")
    if os.path.lexists(args.out):
        parser.error(f"Refusing to overwrite existing output: {args.out}")
    if args.manifest.parent.resolve() in args.out.resolve().parents:
        parser.error("Output must be outside the source manifest directory")
    if (args.latent_chunk < 1 or not math.isfinite(args.latent_step_seconds)
            or args.latent_step_seconds <= 0 or not math.isfinite(args.default_fps)
            or args.default_fps <= 0):
        parser.error("Sampling parameters must be finite and positive")
    # Support direct execution from any working directory without copying code.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rynnvla.datasets.vla_datasets.latent_pretrain import build_index

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # build_index builds the npz inside its own TemporaryDirectory and copies the finished file
    # to --out, because np.savez writes a zip that needs seek-on-write and fuse mounts return
    # OSError(95) for that. Do NOT redirect tempdir to --out's parent: that puts the temporary
    # ZIP back on the same fuse mount the protection exists to avoid. The system default is
    # local disk, which is what makes the copy-based path work.
    meta = build_index(str(args.manifest), str(args.out), args.latent_chunk,
                       args.latent_step_seconds, args.default_fps)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
