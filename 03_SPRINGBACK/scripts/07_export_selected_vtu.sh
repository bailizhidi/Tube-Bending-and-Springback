#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/hpc_common.sh"
cd "$PROJECT_DIR"
PY="${PYTHON_BIN:-$(command -v python)}"
CKPT="${CKPT:-$RUNS_DIR/AW_SpringMGN_150/best_model.pt}"
MAX_EXPORT="${MAX_EXPORT:-15}"
"$PY" predict.py --ckpt "$CKPT" --split test --out_dir "$PRED_DIR/AW_SpringMGN_150_test_vtu" --export_vtu --max_export "$MAX_EXPORT"
