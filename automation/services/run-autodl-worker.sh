#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/automation/config/autodl.toml}"

exec python3 "${REPO_ROOT}/automation/autodl_worker.py" --config "${CONFIG_PATH}"
