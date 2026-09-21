#!/bin/bash
#SBATCH -J ysy_final_stats
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o logs/slurm-stats-%j.out
#SBATCH -e logs/slurm-stats-%j.err
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_OVERRIDE="$PROJECT_DIR"
source "$PROJECT_DIR/slurm_common.sh"
prepare_anaresid_env 1 final_bending_stats
export OMP_NUM_THREADS=8
start_gpu_monitor "logs/gpu_stats_${SLURM_JOB_ID}.csv"
cd "$PROJECT_DIR"
"$PYTHON_BIN" -u compute_final_stats.py --workspace dataeff_workspace --device cuda --theta-final-deg 180 --resume 0
