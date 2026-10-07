#!/usr/bin/env bash
# Launch RynnLAM Stage-1 latent-action training under torchrun.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Paths inside the YAML are relative to your current working directory.
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  "$ROOT/scripts/train_rynnlam.py" --config "$ROOT/rynnlam/configs/stage1.yaml" "$@"
