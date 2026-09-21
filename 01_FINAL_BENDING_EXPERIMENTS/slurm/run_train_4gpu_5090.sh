#!/bin/bash
#SBATCH -J ysy_final_bend
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -o logs/slurm-train-%j.out
#SBATCH -e logs/slurm-train-%j.err
set -euo pipefail
CFG=${1:?Usage: sbatch --gpus=4 -p gpu_5090 ./slurm/run_train_4gpu_5090.sh conf/<experiment>.yaml}
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_OVERRIDE="$PROJECT_DIR"
source "$PROJECT_DIR/slurm_common.sh"
prepare_anaresid_env 4 final_bending_train
export OMP_NUM_THREADS=8
start_gpu_monitor "logs/gpu_train_${SLURM_JOB_ID}.csv"
cd "$PROJECT_DIR"
"$PYTHON_BIN" -u check_experiment_contracts.py
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 train.py --config-name="$(basename "$CFG" .yaml)"
