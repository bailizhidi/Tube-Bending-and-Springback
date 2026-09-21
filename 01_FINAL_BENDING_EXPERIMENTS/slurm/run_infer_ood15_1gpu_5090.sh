#!/bin/bash
#SBATCH -J ysy_ood15
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -o logs/slurm-ood15-%j.out
#SBATCH -e logs/slurm-ood15-%j.err
set -euo pipefail
CFG=${1:?Usage: sbatch --gpus=1 -p gpu_5090 ./slurm/run_infer_ood15_1gpu_5090.sh conf/<exp>.yaml}
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"; export PROJECT_OVERRIDE="$PROJECT_DIR"
source "$PROJECT_DIR/slurm_common.sh"; prepare_anaresid_env 1 external_ood15
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
start_gpu_monitor "logs/gpu_ood15_${SLURM_JOB_ID}.csv"
cd "$PROJECT_DIR"
"$PYTHON_BIN" -u infer_external.py --config "$CFG" --suite ood15 --checkpoint best \
  --external-cache-dir external_workspace/cache_external25 \
  --baseline-cache-dir dataeff_workspace/cache_views/N120 \
  --out-root external_results --continue-on-error "${CONTINUE_ON_ERROR:-0}"
