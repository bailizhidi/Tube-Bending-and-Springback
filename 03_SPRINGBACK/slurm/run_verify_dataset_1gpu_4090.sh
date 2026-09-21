#!/bin/bash
#SBATCH -J ysy_sb_verify
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -o ./logs/%x-%j.out
#SBATCH -e ./logs/%x-%j.err
set -euo pipefail
PROJECT_DIR="$(readlink -f "${SLURM_SUBMIT_DIR:?submit from the project directory}")"
source "$PROJECT_DIR/hpc_common.sh"
prepare_springback_env 1 sb_verify 6
cd "$PROJECT_DIR"
./scripts/02_verify_dataset.sh
