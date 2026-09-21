#!/bin/bash
#SBATCH -J ysy_mesh_diag
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -o logs/slurm-mesh-diag-%j.out
#SBATCH -e logs/slurm-mesh-diag-%j.err
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PROJECT_OVERRIDE="$PROJECT_DIR"
source "$PROJECT_DIR/slurm_common.sh"
prepare_anaresid_env 1 mesh_graph_diagnosis
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
start_gpu_monitor "logs/gpu_mesh_diag_${SLURM_JOB_ID}.csv"
cd "$PROJECT_DIR"
"$PYTHON_BIN" -u diagnose_mesh_graph_stats.py \
  --external-cache-dir external_workspace/cache_external25 \
  --baseline-cache-dir dataeff_workspace/cache_views/N120 \
  --train-stats-dir dataeff_workspace/stats/N120 \
  --frames "${DIAG_FRAMES:-0,45,90,135,179}" \
  --world-radius "${WORLD_RADIUS:-2.0}" \
  --out-dir external_results/mesh_graph_diagnosis
