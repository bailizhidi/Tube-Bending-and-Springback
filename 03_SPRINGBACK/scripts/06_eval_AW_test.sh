#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/hpc_common.sh"
cd "$PROJECT_DIR"
PY="${PYTHON_BIN:-$(command -v python)}"
CKPT="${CKPT:-$RUNS_DIR/AW_SpringMGN_150/best_model.pt}"
OUT="$PRED_DIR/AW_SpringMGN_150_test"
mkdir -p "$OUT"
"$PY" predict.py --ckpt "$CKPT" --split test --out_dir "$OUT" --save_case_npz
"$PY" calc_springback_angle_errors_npz.py --case_dir "$OUT/cases_npz" --out_csv "$OUT/angle_metrics_centerline.csv" --rings 20
"$PY" summarize_results.py --pred_dir "$OUT"
