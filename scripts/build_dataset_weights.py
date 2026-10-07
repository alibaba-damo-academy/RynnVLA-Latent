#!/usr/bin/env python3
"""Derive size-balanced ``dataset_weights`` for a Stage-1 latent mixture.

Uniform sampling over the merged latent index gives every dataset a probability
proportional to its window count, so a corpus dominated by one large source (internet
video, say) spends most of its steps there. This script reweights each dataset by
``windows ** alpha`` and writes a NEW mixture carrying the result:

    alpha  = 1.0  natural proportions exactly -- no ``dataset_weights`` is written
    alpha  < 1.0  flattens the distribution: small sources up, large sources down
    alpha  > 1.0  sharpens it

Window counts use the same formula the dataset itself uses to enumerate samples -- one
sample is one chunk start -- so the weights describe the space training actually walks.
See ``rynnvla/datasets/vla_datasets/latent_pretrain.py`` (``_start_stride`` / ``_ep_starts``)
for the authoritative version; ``tests/test_build_dataset_weights.py`` pins the two together.

The reweighting does NOT change the length of the virtual index: the per-dataset targets
are a largest-remainder rounding of ``w / sum(w) * total``, which sums back to ``total``.
So ``len(dataset)``, and the ``max_steps`` a launcher derives from it, are the same as the
unweighted run -- an alpha arm is comparable to its baseline at equal steps and equal
wall clock. What changes is coverage: a downsampled source no longer visits every window
in one epoch. That is inherent to weighting at a fixed total, not a defect.

Reads an existing mixture rather than taking every field on the command line, so the
weighted arm cannot drift from the baseline it is meant to be compared against.

    python scripts/build_dataset_weights.py \
        --mixture configs/data_latent_pretrain.local.json \
        --out     configs/data_latent_pretrain_a07.json \
        --alpha   0.7
"""
import argparse
import json
import math
from pathlib import Path
import sys

DATA_TYPE = "LatentPretrainDataset"


def _resolve(value):
    """Accept an absolute or CWD-relative path, returning an absolute one.

    Relative is allowed on purpose: the bundled sample mixture and the README's data
    conventions both use repo-relative paths, and rejecting them here would make this
    script the one tool that cannot read the shipped example. Existence and suffix are
    checked by the caller, which has the parser to report through.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _window_counts(index_path, chunk, seconds, parser):
    """Per-dataset chunk-start counts, straight out of the latent index."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rynnvla.datasets.vla_datasets.latent_pretrain import _ACCEPTED_INDEX_VERSIONS

    with np.load(index_path) as z:
        version = int(z["version"][0])
        if version not in _ACCEPTED_INDEX_VERSIONS:
            parser.error(
                f"{index_path} is index version {version}, not one of the accepted "
                f"{_ACCEPTED_INDEX_VERSIONS} -- rebuild it with "
                "scripts/build_latent_pretrain_index.py"
            )
        meta = json.loads(str(z["meta"][0]))
        required = ("ds_names", "ep_ds", "ep_nlat", "ep_stride")
        missing = [key for key in required if key not in z.files]
        if missing:
            parser.error(f"{index_path} lacks {missing}; rebuild it with "
                         "scripts/build_latent_pretrain_index.py")
        if meta.get("latent_chunk") != chunk or not math.isclose(
                float(meta.get("latent_step_seconds", math.nan)), float(seconds)):
            parser.error(
                f"{index_path} was built with latent_chunk={meta.get('latent_chunk')} / "
                f"latent_step_seconds={meta.get('latent_step_seconds')}, but the mixture asks "
                f"for {chunk} / {seconds}. Weights would describe a different sample space "
                "than training walks -- rebuild the index for this recipe."
            )
        ds_names = [str(x) for x in z["ds_names"]]
        ep_ds = z["ep_ds"].astype(np.int64)
        ep_nlat = z["ep_nlat"].astype(np.int64)
        ep_stride = z["ep_stride"].astype(np.int64)

    # Mirrors latent_pretrain.py: start_stride = max(1, stride*chunk//2),
    # span = (chunk-1)*stride + 1, starts = (nlat - span)//start_stride + 1.
    start_stride = np.maximum(1, (ep_stride * chunk) // 2)
    span = (chunk - 1) * ep_stride + 1
    starts = ((ep_nlat - span) // start_stride) + 1
    # build_index drops every episode too short for one window, so starts >= 1 always holds.
    # A non-positive value here means the index and the recipe disagree, which the meta check
    # above should already have caught -- refuse rather than clamp it away and silently
    # under-count a dataset.
    if (starts < 1).any():
        bad = int(np.nonzero(starts < 1)[0][0])
        parser.error(
            f"episode {bad} yields {int(starts[bad])} windows (nlat={int(ep_nlat[bad])}, "
            f"stride={int(ep_stride[bad])}, chunk={chunk}); expected at least 1. The index was "
            "not built for this recipe -- rebuild it."
        )
    windows = np.zeros(len(ds_names), dtype=np.float64)
    np.add.at(windows, ep_ds, starts.astype(np.float64))
    return ds_names, windows


def _report(ds_names, windows, weights):
    """Print the natural -> effective share table, flagging heavy upsampling."""
    total_windows = float(windows.sum())
    wsum = sum(weights.values())
    print(f"[weights] per-dataset windows total {total_windows:,.0f}; "
          f"effective rate = windows**alpha / sum; x = upsample vs natural")
    for di in sorted(range(len(ds_names)), key=lambda i: -weights[ds_names[i]]):
        name = ds_names[di]
        natural = windows[di] / total_windows
        effective = weights[name] / wsum
        ratio = effective / natural if natural > 0 else float("inf")
        flag = "  <-- high upsample: this source is revisited often, watch for overfitting" \
            if ratio > 8 else ""
        print(f"[weights]   {name:24s} nat {natural * 100:6.3f}% -> "
              f"eff {effective * 100:6.3f}%  (x{ratio:5.2f}){flag}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixture", required=True, type=Path,
                        help="existing Stage-1 mixture to derive the weighted arm from")
    parser.add_argument("--out", required=True, type=Path,
                        help="destination mixture; must not exist yet")
    parser.add_argument("--alpha", required=True, type=float,
                        help="temperature exponent on per-dataset window counts "
                             "(1.0 = natural proportions, <1 flattens)")
    parser.add_argument("--index", type=Path, default=None,
                        help="latent index to count windows in "
                             "(default: the mixture entry's data_path)")
    args = parser.parse_args(argv)

    mixture_path = _resolve(args.mixture)
    out_path = _resolve(args.out)
    if not mixture_path.is_file():
        parser.error(f"--mixture does not exist: {mixture_path}")
    if out_path.suffix != ".json":
        parser.error("--out must end in .json")
    if out_path.exists():
        # Deliberately no --force: a mixture feeds data_mixture, which is resume-critical, so
        # silently replacing one that a queued or running job already read is how an arm ends
        # up irreproducible. Pick a new name for a new weighting.
        parser.error(f"Refusing to overwrite existing mixture: {out_path}")
    if out_path.resolve() == mixture_path.resolve():
        parser.error("--out must differ from --mixture")
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and > 0 (1.0 = natural proportions)")

    try:
        entries = json.loads(mixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parser.error(f"--mixture is not valid JSON: {exc}")
    if not isinstance(entries, list):
        parser.error("--mixture must be a JSON list of dataset entries")
    matches = [i for i, e in enumerate(entries)
               if isinstance(e, dict) and e.get("data_type") == DATA_TYPE]
    if len(matches) != 1:
        parser.error(f"expected exactly one {DATA_TYPE} entry in --mixture, found {len(matches)}; "
                     "a weighted arm reweights one latent corpus, not several")
    position = matches[0]
    entry = dict(entries[position])

    chunk = int(entry.get("latent_chunk", 6))
    seconds = float(entry.get("latent_step_seconds", 0.25))
    if args.index is not None:
        index_path = _resolve(args.index)
    elif entry.get("data_path"):
        index_path = _resolve(entry["data_path"])
    else:
        parser.error("the mixture entry has no data_path; pass --index explicitly")
    if not index_path.is_file():
        parser.error(f"latent index does not exist: {index_path}")
    if index_path.suffix != ".npz":
        parser.error(f"latent index must be a .npz built by build_latent_pretrain_index.py: "
                     f"{index_path}")

    ds_names, windows = _window_counts(index_path, chunk, seconds, parser)
    if (windows <= 0).any():
        empty = [ds_names[i] for i in range(len(ds_names)) if windows[i] <= 0]
        parser.error(f"zero windows for {empty}; cannot weight a source with no samples")

    out_entries = [dict(e) if isinstance(e, dict) else e for e in entries]
    out_entry = out_entries[position]
    if math.isclose(args.alpha, 1.0, rel_tol=0.0, abs_tol=0.0):
        # alpha=1 reproduces natural proportions, so write no dataset_weights at all. That
        # keeps the payload identical to an unweighted run instead of routing it through the
        # weighted index for no effect.
        out_entry.pop("dataset_weights", None)
        print(f"[weights] alpha=1.0 reproduces natural proportions exactly: writing no "
              f"dataset_weights, so {out_path.name} samples like the unweighted mixture.")
    else:
        weights = {ds_names[i]: float(windows[i]) ** args.alpha for i in range(len(ds_names))}
        print(f"[weights] alpha={args.alpha} over {len(ds_names)} datasets in {index_path}")
        _report(ds_names, windows, weights)
        out_entry["dataset_weights"] = weights

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_entries, indent=2) + "\n", encoding="utf-8")
    print(f"[weights] wrote {out_path}")
    print(f"[weights] virtual index total is unchanged by weighting, so len(dataset) and any "
          f"max_steps derived from it match the unweighted mixture; pass this file as "
          f"--data-mixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
