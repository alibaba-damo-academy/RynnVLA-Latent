#!/usr/bin/env bash
# One-command check that a fresh clone actually runs, using only the data bundled in data/.
#
#   bash scripts/smoke.sh                       # data + LAM stages (no external weights needed)
#   bash scripts/smoke.sh --src <backbone dir>  # also run the VLA Stage-1 stage
#   bash scripts/smoke.sh --only data           # just the CPU dataset check
#
# Stages, in order:
#   data  verify_sample_data.py -- reads data/sample through the real Stage-1 dataset on CPU
#   lam   generate data/sample_lam if absent, then 2 optimizer steps of RynnLAM Stage-1
#   vla   build a 44M random-weight tiny backbone from --src, then 2 steps of VLA Stage-1
#
# The `vla` stage needs a local Qwen3-VL-family checkpoint for its tokenizer/processor assets
# (RynnBrain-2B or Qwen3-VL-2B-Instruct). Those assets are not redistributed here, so without
# --src the stage is skipped rather than failed. Everything else runs offline with no download.
#
# Outputs land in runs/ (gitignored). Losses are meaningless -- the corpus is synthetic; the
# point is that every stage exits 0 and writes a checkpoint.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SRC=""
ONLY=""
SKIPPED=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)  SRC="${2:?--src needs a directory}"; shift 2 ;;
    --only) ONLY="${2:?--only needs data|lam|vla}"; shift 2 ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "smoke.sh: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done

case "$ONLY" in ""|data|lam|vla) ;; *)
  echo "smoke.sh: --only must be data, lam or vla (got '$ONLY')" >&2; exit 2 ;;
esac

stage_selected() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
banner() { printf '\n=== %s ===\n' "$1"; }

banner "environment"
"$PYTHON" - <<'PY'
import importlib, sys
print(f"python {sys.version.split()[0]}  executable {sys.executable}")
for name in ("torch", "transformers", "numpy", "safetensors", "deepspeed", "flash_attn"):
    try:
        mod = importlib.import_module(name)
        print(f"  {name:14s} {getattr(mod, '__version__', '?')}")
    except ImportError as exc:
        print(f"  {name:14s} MISSING ({exc.__class__.__name__})")
import torch
print(f"  cuda available  {torch.cuda.is_available()}  devices {torch.cuda.device_count()}")
PY

if stage_selected data; then
  banner "stage: data (CPU, no backbone)"
  "$PYTHON" scripts/verify_sample_data.py --mixture data/sample/sample_mixture.json
fi

if stage_selected lam; then
  banner "stage: lam (RynnLAM Stage-1, 2 optimizer steps)"
  if [[ ! -f data/sample_lam/manifest_0000.json ]]; then
    echo "-- generating data/sample_lam"
    "$PYTHON" scripts/make_sample_lam_data.py
  fi
  "$PYTHON" scripts/train_rynnlam.py --config rynnlam/configs/smoke.yaml
  echo "-- lam checkpoints:"
  find runs/smoke_lam -name '*.pt' | sort | sed 's/^/     /'
fi

if stage_selected vla; then
  banner "stage: vla (VLA Stage-1, 2 optimizer steps)"
  if [[ ! -f data/smoke_tiny_model/config.json && -z "$SRC" ]]; then
    cat >&2 <<'MSG'
-- skipped: no tiny backbone and no --src given.
   Build one from any local Qwen3-VL-family checkpoint, then re-run:
     bash scripts/smoke.sh --src /path/to/RynnBrain-2B
MSG
    # Recorded and carried to the summary rather than `exit 0` here: exiting terminated the
    # script before the "smoke complete" banner, so a run that skipped this stage was
    # indistinguishable from one that never printed its summary at all.
    SKIPPED="vla (no tiny backbone; re-run with --src /path/to/RynnBrain-2B)"
  else
    if [[ ! -f data/smoke_tiny_model/config.json ]]; then
      echo "-- building data/smoke_tiny_model from $SRC"
      "$PYTHON" scripts/make_smoke_tiny_model.py --src "$SRC"
    fi
    rm -rf runs/smoke_vla
    # "$PYTHON" -m torch.distributed.run, not a bare `torchrun`: torchrun resolves from PATH and
    # so can belong to a different environment than the interpreter every other stage uses,
    # which turns PYTHON=<venv>/bin/python into a silently mixed-environment run.
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 -m rynnvla.api.train \
      --config rynnvla/configs/stage1_smoke_tiny.json \
      --model_path data/smoke_tiny_model \
      --data_mixture data/sample/sample_mixture.json \
      --output_dir runs/smoke_vla
    echo "-- vla checkpoints:"
    find runs/smoke_vla -maxdepth 1 -mindepth 1 | sort | sed 's/^/     /'
  fi
fi

banner "smoke complete"
if [[ -n "$SKIPPED" ]]; then
  echo "Every stage that ran exited 0. Skipped: $SKIPPED"
else
  echo "Every selected stage exited 0. Artifacts are under runs/ (gitignored)."
fi
