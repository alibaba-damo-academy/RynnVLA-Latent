#!/usr/bin/env bash
# No implicit environment setup, package installation or directory changes.
set -euo pipefail
exec "${PYTHON_BIN:-python}" -m rynnvla.api.launch "$@"
