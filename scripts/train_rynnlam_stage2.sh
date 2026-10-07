#!/usr/bin/env bash
# Launch RynnLAM Stage-2 training under torchrun. Supply --resume PATH; the trainer
# rejects full finetuning without it. Use --no-reset-optimizer to recover an
# interrupted stage-2 run.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  "$ROOT/scripts/train_rynnlam.py" --config "$ROOT/rynnlam/configs/stage2.yaml" "$@"
