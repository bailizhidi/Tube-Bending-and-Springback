#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/hpc_common.sh"
cd "$PROJECT_DIR"
PY="${PYTHON_BIN:-$(command -v python)}"
workers="${NUM_WORKERS:-8}"
"$PY" prepare_pt_dataset.py \
  --config conf/config_AW_150.yaml \
  --data_dir "$SPRINGBACK_PAIR_DIR" \
  --usable_files "$INSPECT_DIR/usable_files.txt" \
  --processed_dir "$GEOM_CACHE_DIR" \
  --num_workers "$workers" --overwrite
