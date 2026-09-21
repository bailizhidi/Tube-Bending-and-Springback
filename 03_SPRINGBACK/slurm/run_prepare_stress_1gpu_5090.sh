#!/bin/bash
#SBATCH -J ysy_sb_cache_stress
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o ./logs/%x-%j.out
#SBATCH -e ./logs/%x-%j.err
set -euo pipefail
PROJECT_DIR="$(readlink -f "${SLURM_SUBMIT_DIR:?submit from the project directory}")"
source "$PROJECT_DIR/hpc_common.sh"
prepare_springback_env 1 sb_cache_stress 8
start_gpu_monitor "$PROJECT_DIR/logs/gpu_cache_stress_${SLURM_JOB_ID}.csv"
cd "$PROJECT_DIR"
NUM_WORKERS=8 ./scripts/03b_prepare_stress_peeq_cache.sh
