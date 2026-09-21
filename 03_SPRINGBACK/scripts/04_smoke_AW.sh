#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/hpc_common.sh"
cd "$PROJECT_DIR"
PY="${PYTHON_BIN:-$(command -v python)}"
"$PY" train_ddp_cached.py --config conf/config_AW_150.yaml \
  --processed_dir "$GEOM_CACHE_DIR" \
  --output_dir "$RUNS_DIR/smoke_AW_150" \
  --epochs 1 --max_steps_per_epoch 2 --val_max_batches 2
