#!/usr/bin/env bash
#SBATCH -J ysy_pod_coeff
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/slurm-pod-coeff-%j.out
#SBATCH -e logs/slurm-pod-coeff-%j.err
set -euo pipefail
REP="${1:?representation required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_multifield_150_npz_compact}"
mkdir -p "$PROJECT_DIR/artifacts"; cd "$PROJECT_DIR"; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u extract_coeff_dataset.py --data-dir "$DATA_DIR" --basis "artifacts/pod_basis_${REP}_train120_grid64x28_q256.npz" --representation "$REP" --output "artifacts/pod_coeff_${REP}_rank32.npz" --max-rank 32
