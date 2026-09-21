#!/bin/bash
#SBATCH -J ysy_final_smoke
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o logs/slurm-smoke-%j.out
#SBATCH -e logs/slurm-smoke-%j.err
set -euo pipefail
CFG=${1:?Usage: sbatch --gpus=1 -p gpu_5090 ./slurm/run_smoke_1gpu_5090.sh conf/<experiment>.yaml}
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_OVERRIDE="$PROJECT_DIR"
source "$PROJECT_DIR/slurm_common.sh"
prepare_anaresid_env 1 final_bending_smoke
export OMP_NUM_THREADS=8
start_gpu_monitor "logs/gpu_smoke_${SLURM_JOB_ID}.csv"
cd "$PROJECT_DIR"
"$PYTHON_BIN" -u check_experiment_contracts.py
"$PYTHON_BIN" -u smoke_test.py --config "$CFG" --steps 2
