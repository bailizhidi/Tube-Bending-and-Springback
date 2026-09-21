#!/bin/bash
#SBATCH -J ysy_bend_exp04_inf
#SBATCH -n 1
#SBATCH --cpus-per-gpu=6
#SBATCH -o logs/slurm-exp04-test-infer-%j.out
#SBATCH -e logs/slurm-exp04-test-infer-%j.err
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_OVERRIDE="$PROJECT_DIR"
source "$PROJECT_DIR/slurm_common.sh"
prepare_anaresid_env 1 final_bending_exp04_test_4090
export OMP_NUM_THREADS=6
start_gpu_monitor "logs/gpu_exp04_test_infer_${SLURM_JOB_ID}.csv"
cd "$PROJECT_DIR"
"$PYTHON_BIN" -u check_experiment_contracts.py
"$PYTHON_BIN" -u inference.py \
  --config conf/exp04_local_anaresid_x10_2step.yaml \
  --checkpoint best \
  --split test
