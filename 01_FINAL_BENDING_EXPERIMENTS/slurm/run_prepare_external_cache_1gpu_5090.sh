#!/bin/bash
#SBATCH -J ysy_ext_cache
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -o logs/slurm-ext-cache-%j.out
#SBATCH -e logs/slurm-ext-cache-%j.err
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_OVERRIDE="$PROJECT_DIR"
source "$PROJECT_DIR/slurm_common.sh"
prepare_anaresid_env 1 external_cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$PROJECT_DIR"
DATASET_DIR="${EXTERNAL_DATASET_DIR:-/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_external25_npz_compact}"
"$PYTHON_BIN" -u prepare_external_cache.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir external_workspace/cache_external25 \
  --train-stats-dir dataeff_workspace/stats/N120 \
  --num-time-steps 181 --world-radius 2.0 --overwrite "${OVERWRITE:-0}"
