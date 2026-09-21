#!/usr/bin/env bash
#SBATCH -J ysy_pod_train
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/slurm-pod-train-%j.out
#SBATCH -e logs/slurm-pod-train-%j.err
set -euo pipefail
REP="${1:?representation required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$PROJECT_DIR/artifacts"; cd "$PROJECT_DIR"; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u train_rom_mlp.py --dataset "artifacts/pod_coeff_${REP}_rank32.npz" --output "artifacts/pod_rom_${REP}_rank16_process.pt" --representation "$REP" --rank 16 --width 128 --depth 3 --epochs 800 --batch-size 512 --lr 1e-3 --weight-decay 0 --patience 120 --seed 0
