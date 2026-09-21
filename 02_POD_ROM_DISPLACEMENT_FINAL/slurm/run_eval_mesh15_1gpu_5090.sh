#!/usr/bin/env bash
#SBATCH -J ysy_pod_mesh15
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -o logs/slurm-pod-mesh15-%j.out
#SBATCH -e logs/slurm-pod-mesh15-%j.err
set -euo pipefail
REP="${1:-local_residual}"
PROJECT_DIR="${POD_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
BASE_DATA="${BASE_DATA_DIR:-/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_multifield_150_npz_compact}"
EXT_DATA="${EXTERNAL_DATA_DIR:-/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_external25_npz_compact}"
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results_external/mesh15/$REP"; cd "$PROJECT_DIR"; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u eval_mesh15.py --base-data-dir "$BASE_DATA" --external-data-dir "$EXT_DATA" \
 --basis "artifacts/pod_basis_${REP}_train120_grid64x28_q256.npz" \
 --dataset "artifacts/pod_coeff_${REP}_rank32.npz" \
 --checkpoint "artifacts/pod_rom_${REP}_rank16_process.pt" \
 --representation "$REP" --out-dir "results_external/mesh15/$REP"
