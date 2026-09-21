#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/hpc_common.sh"
cd "$PROJECT_DIR"
PY="${PYTHON_BIN:-$(command -v python)}"
print_hpc_paths
"$PY" inspect_pairs.py --dataset_dir "$SPRINGBACK_PAIR_DIR" --out_dir "$INSPECT_DIR"
"$PY" check_contract.py --config conf/config_AW_150.yaml --pair_dir "$SPRINGBACK_PAIR_DIR"
echo "[VERIFY] PASSED"
