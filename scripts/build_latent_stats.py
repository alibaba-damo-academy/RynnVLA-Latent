#!/usr/bin/env python3
"""Compute latent normalization statistics (mean/std) over a labeled corpus.

Stage-1 reads these through ``latent_loader.load_latent_stats``, which expects a JSON object
with ``mean`` and ``std`` arrays whose length equals the recipe's ``latent_action_dim``.
Nothing else in the pipeline produces that file: the ``<Dataset>_stats.json`` files beside the
RynnVLA-Base manifests are dataset inventory stats (episodes / views / flow_rate), not latent
moments, and pointing ``latent_stats_path`` at one of them trips the width assertion.

Without a stats file the dataset silently falls back to identity (0/1) normalization and trains
on raw latents, so this is a required step, not an optional refinement.

Statistics are accumulated in one streaming pass (sum / sum_sq / min / max / count) so memory
stays flat regardless of corpus size. ``std`` is the unbiased (n-1) form, matching
``base.py:_finalize_leaf``. The manifest is streamed too, in either format (JSON array or the
JSONL that ``scripts/rebuild_latent_manifest.py`` writes): a full-corpus manifest is far too
large to ``json.load``, which materializes the whole parse tree and costs GBs of RSS per
million episodes.

``--sample-rate`` reads a deterministic subset of episodes instead of all of them. Full-corpus
stats mean reading every latent array -- one npz per (episode, view), hundreds of KB each --
which is hours of fuse IO for numbers that a small sample estimates to well under a per-mille
of relative error.
Selection is a hash of ``(dataset, episode_id)``, so it is stable across runs, processes and
shardings -- never ``hash()``, whose salt would make two runs disagree. Sampling is uniform over
episodes, which is proportional to episode count per dataset: that matches training, where a
single-source mixture draws episodes uniformly. The rate is recorded in the output JSON.

Usage:
    python scripts/build_latent_stats.py \
        --manifest /path/to/vla_train_index_manifest.json \
        --out /path/to/latent_stats.json \
        --latent-dim 608
    # full corpus, 1% of episodes, 64 readers:
    python scripts/build_latent_stats.py --manifest /path/to/rynnvla_base.jsonl \
        --out /path/to/latent_stats.json --latent-dim 608 \
        --sample-rate 0.01 --workers 64
"""
import argparse
import functools
import itertools
import json
import os
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import blake2b
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rynnvla.utils.manifest_io import iter_manifest  # noqa: E402


def episode_selected(dataset: str, episode_id: str, rate: float) -> bool:
    """Deterministic per-episode sampling decision (see the module docstring)."""
    if rate >= 1.0:
        return True
    digest = blake2b(f"{dataset}\0{episode_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") < rate * (1 << 64)


def iter_latent_paths(manifest_path: Path, sample_rate: float = 1.0, tally: Counter = None):
    """Yield the latent paths to read, streaming the manifest one episode at a time.

    Duplicates are dropped within an episode (two views pointing at one file must not
    double-count its rows) but not across episodes: a manifest that lists the same episode
    twice would skew the moments slightly, and ``rebuild_latent_manifest.py`` already refuses
    to emit one. A global ``seen`` set would need gigabytes at corpus scale, which is the
    memory this streaming exists to avoid.
    """
    tally = Counter() if tally is None else tally
    for entry in iter_manifest(str(manifest_path)):
        tally["episodes"] += 1
        dataset = str(entry.get("dataset") or "")
        episode = str(entry.get("episode_id") or "")
        if not episode_selected(dataset, episode, sample_rate):
            continue
        tally["episodes_sampled"] += 1
        seen = set()
        for view in entry.get("views") or []:
            path = view.get("latent_path")
            if path and path not in seen:
                seen.add(path)
                yield path


def load_latent_action(path: str) -> np.ndarray:
    """Read the [T, dim] latent_action array. allow_pickle stays off -- these files hold
    float16 latents plus an int32 pair-index array and a fixed-width unicode meta scalar,
    none of which need object deserialization."""
    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file

        data = load_file(path)
        if "latent_action" not in data:
            raise KeyError(f"{path}: expected key 'latent_action', found {sorted(data)[:8]}")
        return data["latent_action"]
    with np.load(path, allow_pickle=False) as archive:
        return archive["latent_action"]


def accumulate_one(path: str, dim: int):
    """Per-file partial reduction, safe to run in a thread pool.

    Returns its own local ``(sum, sum_sq, min, max, count)`` rather than mutating shared state,
    so merging is just per-dimension min/max plus sum and count addition. The n-1 standard
    deviation is computed only after every partial is combined.
    """
    if not os.path.isfile(path):
        return ("missing", path, None)
    try:
        arr = load_latent_action(path).astype(np.float64)
    # zipfile.BadZipFile subclasses Exception directly, so it is caught by neither OSError nor
    # ValueError. That is exactly what a latent.npz truncated by an interrupted labeling run
    # raises, and without it one corrupt file among millions aborts an hours-long pass with no
    # partial output -- even though rebuild_latent_manifest.probe_latent already classified the
    # same file as `unreadable_latent` and dropped it one stage earlier in the same pipeline.
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return ("missing", path, None)
    if arr.ndim != 2 or arr.shape[1] != dim:
        # Fail loudly rather than averaging incompatible representations together: a corpus
        # labelled with one representation and a --latent-dim for another would otherwise
        # produce stats that silently mismatch the arrays at training time.
        return ("wrong_width", path, tuple(arr.shape))
    flat = arr.reshape(-1, dim)
    return ("ok", None, (flat.sum(axis=0), (flat ** 2).sum(axis=0),
                         flat.min(axis=0), flat.max(axis=0), int(flat.shape[0])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read this many latent files (smoke subsets).")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="Fraction of episodes to read, selected by a stable hash of "
                             "(dataset, episode_id). 1.0 reads everything.")
    parser.add_argument("--workers", type=int, default=16,
                        help="Concurrent file readers. The corpus lives on a fuse mount, so "
                             "this is latency-bound: serial over ~800k files takes hours.")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="Paths submitted to the pool at once. Executor.map over an "
                             "unbounded generator would queue ~20M futures.")
    parser.add_argument("--report-every", type=int, default=200000,
                        help="Print a progress line after this many files (0 disables).")
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        parser.error(f"Manifest does not exist: {args.manifest}")
    if args.out.suffix != ".json":
        parser.error("--out must end in .json")
    if os.path.lexists(args.out):
        parser.error(f"Refusing to overwrite existing output: {args.out}")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.latent_dim < 1:
        parser.error("--latent-dim must be positive")
    if not 0.0 < args.sample_rate <= 1.0:
        parser.error("--sample-rate must be in (0, 1]")

    dim = args.latent_dim
    total = np.zeros(dim, np.float64)
    total_sq = np.zeros(dim, np.float64)
    vmin = np.full(dim, np.inf, np.float64)
    vmax = np.full(dim, -np.inf, np.float64)
    count = 0
    read = 0
    missing = 0
    wrong_width = 0
    tally = Counter()
    started = time.perf_counter()
    next_report = args.report_every

    paths = iter_latent_paths(args.manifest, args.sample_rate, tally)
    if args.limit is not None:
        paths = itertools.islice(paths, args.limit)

    def merge(batch):
        """Read one batch and fold it into the running moments, in submission order.

        Ordered merging is what keeps a 64-worker run bitwise identical to a serial one:
        ``pool.map`` yields results in submission order no matter how the reads finished.
        """
        nonlocal total, total_sq, vmin, vmax, count, read, missing, wrong_width
        if not batch:
            return
        for status, path, payload in pool.map(functools.partial(accumulate_one, dim=dim), batch):
            if status == "missing":
                missing += 1
            elif status == "wrong_width":
                wrong_width += 1
                if wrong_width == 1:
                    print(f"[stats] width mismatch: {path} has shape {payload}, "
                          f"expected [T, {dim}]")
            else:
                part_sum, part_sum_sq, part_min, part_max, part_count = payload
                total += part_sum
                total_sq += part_sum_sq
                vmin = np.minimum(vmin, part_min)
                vmax = np.maximum(vmax, part_max)
                count += part_count
                read += 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        batch = []
        for path in paths:
            batch.append(path)
            if len(batch) < args.batch_size:
                continue
            merge(batch)
            batch = []
            done = read + missing
            if args.report_every and done >= next_report:
                next_report += args.report_every
                elapsed = time.perf_counter() - started
                print(f"[stats] {done:,} files ({read:,} read, {missing:,} missing), "
                      f"{count:,} rows, {tally['episodes_sampled']:,} episodes, "
                      f"{done / elapsed:,.0f} files/s, elapsed {elapsed / 60:,.1f} min",
                      flush=True)
        merge(batch)

    if wrong_width:
        raise SystemExit(
            f"[stats] {wrong_width} latent file(s) had the wrong width for --latent-dim={dim}; "
            "the corpus representation and the requested dimension disagree. No stats written."
        )
    if count < 2:
        raise SystemExit(
            f"[stats] only {count} latent rows were usable (read={read}, missing={missing}); "
            "cannot estimate a standard deviation. No stats written."
        )

    mean = total / count
    var = np.maximum(0.0, (total_sq - total ** 2 / count) / (count - 1))
    std = np.sqrt(var)
    # A zero standard deviation would divide by zero during normalization. It means a latent
    # channel is constant across the whole corpus, which is a labeling problem, not something
    # to paper over with an epsilon.
    if not bool(np.all(std > 0)):
        bad = int(np.argmin(std))
        raise SystemExit(
            f"[stats] std[{bad}] = {std[bad]}: latent channel {bad} is constant across "
            f"{count} rows. Fix the labeling before normalizing. No stats written."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump({
            "latent_dim": dim,
            "rows": count,
            "files": read,
            "missing": missing,
            # Provenance: someone reading this file later must be able to tell that the
            # normalization constants come from a 1% sample rather than the whole corpus.
            "sample_rate": args.sample_rate,
            "episodes": tally["episodes"],
            "episodes_sampled": tally["episodes_sampled"],
            "manifest": str(args.manifest),
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "min": vmin.astype(np.float32).tolist(),
            "max": vmax.astype(np.float32).tolist(),
        }, handle)

    print(json.dumps({
        "out": str(args.out), "latent_dim": dim, "rows": count, "files": read,
        "missing": missing, "sample_rate": args.sample_rate,
        "episodes": tally["episodes"], "episodes_sampled": tally["episodes_sampled"],
        "seconds": round(time.perf_counter() - started, 1),
        "std_min": float(std.min()), "std_max": float(std.max()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
