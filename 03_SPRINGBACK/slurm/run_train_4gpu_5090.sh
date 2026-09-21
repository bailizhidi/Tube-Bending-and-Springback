#!/bin/bash
#SBATCH -J ysy_sb_train
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -o ./logs/%x-%j.out
#SBATCH -e ./logs/%x-%j.err
set -euo pipefail
CFG="${1:?Usage: sbatch --gpus=4 -p gpu_5090 ./slurm/run_train_4gpu_5090.sh conf/<config>.yaml [run_name]}"
RUN_NAME="${2:-$(basename "$CFG" .yaml)}"
PROJECT_DIR="$(readlink -f "${SLURM_SUBMIT_DIR:?submit from the project directory}")"
source "$PROJECT_DIR/hpc_common.sh"
prepare_springback_env 4 sb_train_5090 8
start_gpu_monitor "$PROJECT_DIR/logs/gpu_${RUN_NAME}_${SLURM_JOB_ID}.csv"
cd "$PROJECT_DIR"
NGPU=4 ./scripts/05_train_from_config.sh "$CFG" "$RUN_NAME"
