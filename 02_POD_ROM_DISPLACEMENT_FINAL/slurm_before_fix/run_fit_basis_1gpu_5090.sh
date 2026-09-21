#!/usr/bin/env bash
#SBATCH -J ysy_pod_fit
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/slurm-pod-fit-%j.out
#SBATCH -e logs/slurm-pod-fit-%j.err
set -euo pipefail
REP="${1:?usage: $0 direct|global_residual|local_residual}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_multifield_150_npz_compact}"
mkdir -p "$PROJECT_DIR/artifacts" "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u fit_pod_basis.py --data-dir "$DATA_DIR" --representation "$REP" --output "artifacts/pod_basis_${REP}_train120_grid64x28_q256.npz" --n-s 64 --n-phi 28 --n-modes 256 --niter 5 --seed 0
