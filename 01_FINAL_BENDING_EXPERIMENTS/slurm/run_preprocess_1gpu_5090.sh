#!/bin/bash
#SBATCH -J ysy_final_preprocess
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o logs/slurm-preprocess-%j.out
#SBATCH -e logs/slurm-preprocess-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_OVERRIDE="$PROJECT_DIR"

source "$PROJECT_DIR/slurm_common.sh"

prepare_anaresid_env 1 final_bending_preprocess

export OMP_NUM_THREADS=8

start_gpu_monitor "logs/gpu_preprocess_${SLURM_JOB_ID}.csv"

cd "$PROJECT_DIR"

"$PYTHON_BIN" -u preprocess_shared_raw_x10_v2.py \
  --dataset-dir /data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_multifield_150_npz_compact \
  --output-dir shared_raw_packed_clean150 \
  --num-time-steps 181 \
  --world-radius 2.0 \
  --overwrite 1
