#!/usr/bin/env bash
#SBATCH -J ysy_pod_eval
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/slurm-pod-eval-%j.out
#SBATCH -e logs/slurm-pod-eval-%j.err
set -euo pipefail
REP="${1:?representation required}"; SPLIT="${2:-val}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_multifield_150_npz_compact}"
mkdir -p "$PROJECT_DIR/results/$REP/$SPLIT"; cd "$PROJECT_DIR"; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u evaluate_rom.py --data-dir "$DATA_DIR" --basis "artifacts/pod_basis_${REP}_train120_grid64x28_q256.npz" --dataset "artifacts/pod_coeff_${REP}_rank32.npz" --checkpoint "artifacts/pod_rom_${REP}_rank16_process.pt" --representation "$REP" --split "$SPLIT" --out-dir "results/$REP/$SPLIT"
