"""Portable torchrun launcher. Importing this module does not import torch."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


PRESETS = {
    "stage1": "stage1_4b.json",
    "stage1_2b": "stage1_2b.json",
    "stage2": "stage2_v5_4b.json",
    "stage2_2b": "stage2_v5_2b.json",
}


def normalize_options(argv):
    """Accept kebab-case and TrainingArguments' original underscore spelling."""
    return [
        "--" + token[2:].split("=", 1)[0].replace("-", "_")
        + ("=" + token.split("=", 1)[1] if "=" in token else "")
        if token.startswith("--") else token
        for token in argv
    ]


def build_command(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch Stage1 or Stage2 V5 (2B/4B) with torchrun.",
        epilog="Other options override TrainingArguments (e.g. --max-steps 2). "
        "Defaults: micro batch 4, accumulation 4, 8 ranks = global batch 128. "
        "Changing the rank count does not automatically rescale the batch or LR.",
        allow_abbrev=False,
    )
    parser.add_argument("--stage", choices=PRESETS, required=True)
    parser.add_argument("--model_path", "--model-path", required=True)
    parser.add_argument("--data_mixture", "--data-mixture", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--nproc_per_node", "--nproc-per-node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", "--node-rank", type=int, default=0)
    parser.add_argument("--master_addr", "--master-addr")
    parser.add_argument("--master_port", "--master-port", type=int, default=29500)
    parser.add_argument("--resume", action="store_true", help="Resume latest complete training checkpoint in output-dir.")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Print command only; no CUDA, files or processes.")
    args, overrides = parser.parse_known_args(normalize_options(sys.argv[1:] if argv is None else argv))
    if args.nnodes < 1 or args.nproc_per_node < 1:
        parser.error("nnodes and nproc-per-node must be positive integers")
    if not 0 <= args.node_rank < args.nnodes:
        parser.error("node-rank must be in [0, nnodes)")
    if not 1 <= args.master_port <= 65535:
        parser.error("master-port must be in [1, 65535]")
    if args.nnodes > 1 and not args.master_addr:
        parser.error("multinode launch requires an explicit --master-addr reachable by every node")
    if any(token.split("=", 1)[0] == "--config" for token in overrides):
        parser.error("launch selects --config via --stage; use api.train directly for a custom JSON recipe")

    preset = Path(__file__).resolve().parents[1] / "configs" / PRESETS[args.stage]
    if not preset.is_file():
        parser.error(f"Installed training preset not found: {preset}")
    command = [sys.executable, "-m", "torch.distributed.run"]
    if args.nnodes == 1 and args.master_addr is None:
        command.append("--standalone")
    else:
        command += ["--master_addr", args.master_addr, "--master_port", str(args.master_port)]
    command += [
        "--nnodes", str(args.nnodes), "--node_rank", str(args.node_rank),
        "--nproc_per_node", str(args.nproc_per_node),
        "--module", "rynnvla.api.train", "--config", str(preset),
        "--model_path", args.model_path, "--data_mixture", args.data_mixture,
        "--output_dir", args.output_dir,
    ]
    if args.resume:
        command.append("--resume")
    command += overrides
    return command, args.dry_run


def main(argv=None):
    command, dry_run = build_command(argv)
    print(shlex.join(command), flush=True)
    if not dry_run:
        from .train import check_output_dir

        resume = "--resume" in command
        # Only node 0 owns the output directory. Every node ran this pre-flight before, but
        # node 0's Trainer creates the directory shortly after starting, so a node that came up
        # a little later saw it already there and aborted with "Output already exists" without
        # ever launching torchrun.
        node_rank = int(command[command.index("--node_rank") + 1])
        try:
            if node_rank == 0:
                check_output_dir(command[command.index("--output_dir") + 1], resume=resume)
            preset = Path(command[command.index("--config") + 1]).name
            if preset in (PRESETS["stage2"], PRESETS["stage2_2b"]) and not resume:
                from .export_checkpoint import checkpoint_metadata

                source_stage = "stage1_2b" if preset == PRESETS["stage2_2b"] else "stage1"
                checkpoint_metadata(command[command.index("--model_path") + 1], stage=source_stage)
        except (OSError, ValueError) as exc:
            print(f"RynnVLA launch failed: {exc}", file=sys.stderr)
            return 2
        return subprocess.call(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
