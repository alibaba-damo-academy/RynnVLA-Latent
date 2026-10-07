#!/usr/bin/env bash
# Public single-checkpoint RynnLAM evaluation entrypoint; all data/output paths explicit.
# Subcommands: extract / merge / regress (regress needs an external --lary-root checkout).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/evaluate_rynnlam.py" "$@"
