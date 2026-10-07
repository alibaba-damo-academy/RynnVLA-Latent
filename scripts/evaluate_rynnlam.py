#!/usr/bin/env python3
"""Validate/merge extraction and run the official external LARYBench MLP probe.

Extraction needs no LARY installation. Regression requires a supplied checkout
with regression/main.py and its dependencies; no registry injection or vendoring.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rynnlam.inference import file_sha256 as sha256_file
from stream_extract_tar import DATASETS, digest, validate_output, write_csv, write_json


def load_report(path, statuses=("complete",)):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if report.get("status") not in statuses:
        raise ValueError(f"Extraction is not complete: {path}")
    if digest(report["protocol"]) != report["protocol_id"]:
        raise ValueError(f"Protocol digest mismatch: {path}")
    if sha256_file(report["csv"]) != report["csv_sha256"]:
        raise ValueError(f"CSV changed since extraction: {path}")
    with open(report["csv"], newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    ids = [int(row["rynnlam_row_id"]) for row in rows]
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(report["completed_rows"]):
        raise ValueError(f"Invalid row coverage: {path}")
    if any(row["rynnlam_protocol_id"] != report["protocol_id"] for row in rows):
        raise ValueError(f"Mixed row protocols: {path}")
    if report["status"] == "complete" and sorted(ids) != list(
        range(report["metadata_rows"])
    ):
        raise ValueError(f"Missing metadata rows: {path}")
    return report, rows, fields


def merge(args):
    loaded = [load_report(p, ("complete", "shard_complete")) for p in args.reports]
    first = loaded[0][0]
    expected = first["protocol"]["num_shards"]
    if len(loaded) != expected or {r[0].get("shard") for r in loaded} != set(
        range(expected)
    ):
        raise ValueError("Provide exactly one report from every shard")
    if any(r[0]["protocol_id"] != first["protocol_id"] for r in loaded):
        raise ValueError("Cannot merge different checkpoints or extraction protocols")
    rows, ids, shapes = [], set(), set()
    for report, shard_rows, fields in loaded:
        current = {int(r["rynnlam_row_id"]) for r in shard_rows}
        if current & ids or current != set(report["owned_rows"]):
            raise ValueError("Duplicate ownership or incomplete shard")
        ids.update(current)
        rows.extend(shard_rows)
        shapes.update(tuple(s) for s in report["token_shapes"])
    if ids != set(range(first["metadata_rows"])):
        missing = sorted(set(range(first["metadata_rows"])) - ids)
        raise ValueError(
            f"Metadata rows missing from all archives: {missing[:20]} (total {len(missing)})"
        )
    if len(shapes) != 1:
        raise ValueError("Inconsistent token dimensions across shards")
    output = (
        Path(args.output_dir).resolve()
        / first["protocol"]["dataset"]
        / first["protocol_id"][:16]
        / first["protocol"]["split"]
    )
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Merge output locked: {lock}") from exc
    try:
        csv_path = output / f"{first['protocol']['split']}_la.csv"
        rows.sort(key=lambda r: int(r["rynnlam_row_id"]))
        write_csv(csv_path, rows, fields)
        report = dict(
            first,
            status="complete",
            shard=None,
            owned_rows=sorted(ids),
            completed_rows=sorted(ids),
            csv=str(csv_path),
            csv_sha256=sha256_file(csv_path),
            token_shapes=[list(next(iter(shapes)))],
            flat_feature_dim=int(np.prod(next(iter(shapes)))),
            merged_reports=[str(Path(p).resolve()) for p in args.reports],
        )
        path = output / "report.json"
        write_json(path, report)
        print(path)
        return path
    finally:
        lock.rmdir()


def robot_stats(train_rows, val_rows, dim):
    # Historical probe protocol: up to 400 train chunks per robot, RandomState(0).
    groups = {}
    for row in train_rows:
        groups.setdefault(row["robot_type"], []).append(row)
    missing = {r["robot_type"] for r in val_rows} - groups.keys()
    if missing:
        raise ValueError(
            f"Validation robot types lack train statistics: {sorted(missing)}"
        )
    rng = np.random.RandomState(0)
    stats = {}
    for robot, rows in groups.items():
        chosen = rng.choice(len(rows), min(400, len(rows)), replace=False)
        actions = np.concatenate(
            [
                np.load(rows[i]["action"], allow_pickle=False).reshape(-1, dim)
                for i in chosen
            ]
        )
        if not np.isfinite(actions).all():
            raise ValueError(f"Non-finite actions for robot {robot}")
        std = actions.std(0)
        stats[robot] = {
            "mean": actions.mean(0).tolist(),
            "std": np.where(std < 1e-6, 1.0, std).tolist(),
        }
    return {
        "robot_stats": stats,
        "sampling": {"seed": 0, "max_chunks_per_robot": 400, "split": "train"},
    }


def regression_command(args, train, val, save_dir, stats_path=None):
    entry = Path(args.lary_root).resolve() / "regression" / "main.py"
    if not entry.is_file():
        raise ValueError(f"Official LARYBench entrypoint not found: {entry}")
    protocol = train["protocol"]
    command = [
        sys.executable,
        str(entry),
        "--train_csv",
        train["csv"],
        "--val_csv",
        val["csv"],
        "--dataset",
        DATASETS[protocol["dataset"]]["lary_name"],
        "--stride",
        str(protocol["stride"]),
        "--model_type",
        "mlp",
        "--hidden_dim",
        str(args.hidden_dim),
        "--num_blocks",
        str(args.num_blocks),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--save_dir",
        str(save_dir),
        # The external LARYBench entrypoint names its own run with this flag; WANDB_MODE
        # below is forced to disabled, so the child process never reaches any service.
        "--wandb_name",
        f"rynnlam-{protocol['dataset']}-{train['protocol_id'][:12]}",
    ]
    if stats_path:
        command.extend(["--global_stats_json", str(stats_path)])
    return command


def regress(args):
    train, train_rows, _ = load_report(args.train_report)
    val, val_rows, _ = load_report(args.val_report)
    if train["protocol"]["split"] != "train" or val["protocol"]["split"] != "val":
        raise ValueError("Expected train and val reports respectively")
    ignored = {"split", "archives", "metadata_sha256", "num_shards"}
    train_protocol = {k: v for k, v in train["protocol"].items() if k not in ignored}
    val_protocol = {k: v for k, v in val["protocol"].items() if k not in ignored}
    if train_protocol != val_protocol or train["token_shapes"] != val["token_shapes"]:
        raise ValueError(
            "Train/validation checkpoint, normalization, geometry or representation mismatch"
        )
    for report, rows in ((train, train_rows), (val, val_rows)):
        for row in rows:
            validate_output(
                row["la_path"],
                row["action"],
                report["protocol_id"],
                int(row["rynnlam_row_id"]),
                report["protocol"]["dataset"],
                report["protocol"]["stride"],
                report["token_shapes"],
            )
    settings = {
        k: getattr(args, k)
        for k in (
            "epochs",
            "lr",
            "seed",
            "hidden_dim",
            "num_blocks",
            "batch_size",
            "num_workers",
        )
    }
    if (
        any(
            settings[k] < 1
            for k in ("epochs", "hidden_dim", "num_blocks", "batch_size")
        )
        or args.lr <= 0
        or args.num_workers < 0
    ):
        raise ValueError("Invalid regression hyperparameters")
    entry = Path(args.lary_root).resolve() / "regression" / "main.py"
    if not entry.is_file():
        raise ValueError(f"Official LARYBench entrypoint not found: {entry}")
    run_protocol = {
        "train": train["protocol_id"],
        "val": val["protocol_id"],
        "settings": settings,
        "lary_main_sha256": sha256_file(entry),
        "lary_root": str(entry.parent.parent),
        "action_normalization": (
            "train per-robot sample400 seed0"
            if train_protocol["dataset"] in ("agibot", "robocoin")
            else "official LARYBench train statistics"
        ),
    }
    run_id = digest(run_protocol)
    out = (
        Path(args.output_root).resolve()
        / train_protocol["dataset"]
        / f"probe-{run_id[:16]}"
    )
    stats_path = (
        out / "robot_stats.json"
        if train_protocol["dataset"] in ("agibot", "robocoin")
        else None
    )
    command = regression_command(args, train, val, out / "checkpoints", stats_path)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return command
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Regression output locked: {lock}") from exc
    try:
        if stats_path:
            write_json(
                stats_path,
                robot_stats(train_rows, val_rows, train_protocol["action_dim"]),
            )
        result = {
            "status": "running",
            "protocol": run_protocol,
            "command": command,
            "train_extraction": train,
            "val_extraction": val,
        }
        write_json(out / "result.json", result)
        env = os.environ.copy()
        # Only the official supplied source checkout enters PYTHONPATH.
        env["PYTHONPATH"] = str(entry.parent.parent)
        env["DATA_DIR"] = str(out)
        env["WANDB_MODE"] = "disabled"
        env["WANDB_DISABLED"] = "true"
        log_path = out / "regression.log"
        with open(log_path, "w") as log:
            process = subprocess.run(
                command,
                cwd=entry.parent.parent,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        text = log_path.read_text(errors="replace")
        # Capture whole tokens: partial numeric matches hide NaN/Inf and typos.
        metric_pattern = r"Val Seen MSE:[ \t]*([^\s|]*)"
        matches = re.findall(metric_pattern, text)
        try:
            scores = [float(s) for s in matches]
            valid_scores = bool(scores) and bool(np.isfinite(scores).all())
        except ValueError:
            scores, valid_scores = [], False
        final_epochs = list(
            re.finditer(rf"Epoch[ \t]+{args.epochs}[ \t]*\|[ \t]*Train Loss", text)
        )
        finished = bool(final_epochs)
        final_scores = []
        if finished:
            # Restrict validation to the last final-epoch block, never an older score.
            final_block = re.split(
                r"Epoch[ \t]+\d+[ \t]*\|[ \t]*Train Loss",
                text[final_epochs[-1].end() :],
                maxsplit=1,
            )[0]
            final_scores = re.findall(metric_pattern, final_block)
        success = (
            process.returncode == 0 and finished and bool(final_scores) and valid_scores
        )
        result.update(
            status="complete" if success else "failed",
            returncode=process.returncode,
            best_val_seen_mse=min(scores) if valid_scores else None,
            completed_final_epoch=finished,
            log=str(log_path),
        )
        write_json(out / "result.json", result)
        if not success:
            raise RuntimeError(f"Regression failed or incomplete; see {log_path}")
        print(f"Best Val-Seen MSE: {min(scores)}; {out / 'result.json'}")
        return out / "result.json"
    finally:
        lock.rmdir()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extraction = sub.add_parser(
        "extract",
        help="Frozen extraction; use stream_extract_tar.py --help for options",
        add_help=False,
    )
    extraction.add_argument("extraction_args", nargs=argparse.REMAINDER)
    merger = sub.add_parser(
        "merge", help="Validate every shard and require complete metadata coverage"
    )
    merger.add_argument("--reports", nargs="+", required=True)
    merger.add_argument("--output-dir", required=True)
    probe = sub.add_parser(
        "regress", help="Run official external LARYBench regression only"
    )
    probe.add_argument("--lary-root", required=True)
    probe.add_argument("--train-report", required=True)
    probe.add_argument("--val-report", required=True)
    probe.add_argument("--output-root", required=True)
    probe.add_argument("--epochs", type=int, default=30)
    probe.add_argument("--hidden-dim", type=int, default=4096)
    probe.add_argument("--num-blocks", type=int, default=2)
    probe.add_argument("--batch-size", type=int, default=64)
    probe.add_argument("--num-workers", type=int, default=4)
    probe.add_argument("--lr", type=float, default=1e-4)
    probe.add_argument("--seed", type=int, default=42)
    probe.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without importing LARY or training",
    )
    return parser


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        from stream_extract_tar import build_parser as extraction_parser, extract

        args = extraction_parser().parse_args(sys.argv[2:])
        operation = extract
    else:
        args = build_parser().parse_args()
        operation = {"merge": merge, "regress": regress}[args.command]
    try:
        operation(args)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
