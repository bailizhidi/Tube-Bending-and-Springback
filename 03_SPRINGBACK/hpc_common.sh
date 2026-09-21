#!/usr/bin/env bash
set -euo pipefail

LOGIN_ROOT="/data/home/scxk573/run/lsz/ysy/physicsnemo"
RUN_ROOT="/data/run01/scxk573/lsz/ysy/physicsnemo"

# User-facing upload location requested for the Windows-extracted springback archive.
export SPRINGBACK_ZIP_DIR="${SPRINGBACK_ZIP_DIR:-${LOGIN_ROOT}/data/zip}"
export SPRINGBACK_PAIR_DIR="${SPRINGBACK_PAIR_DIR:-${LOGIN_ROOT}/data/dataset_springback_pair_Base_TC4_150}"

# Keep the project where the rest of PhysicsNeMo structural-mechanics projects live.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/train_ddp_cached.py" ]]; then
    PROJECT_DIR="$(readlink -f "${SLURM_SUBMIT_DIR}")"
else
    PROJECT_DIR="$(readlink -f "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
fi
export PROJECT_DIR

# Once paths exist, canonicalize /data/home aliases to the compute-node physical path.
canonical_existing() {
    local p="$1"
    if [[ -e "$p" ]]; then readlink -f "$p"; else echo "$p"; fi
}
export SPRINGBACK_PAIR_DIR="$(canonical_existing "$SPRINGBACK_PAIR_DIR")"
export SPRINGBACK_ZIP_DIR="$(canonical_existing "$SPRINGBACK_ZIP_DIR")"

export GEOM_CACHE_DIR="${GEOM_CACHE_DIR:-${PROJECT_DIR}/data/processed_geom_150}"
export STRESS_CACHE_DIR="${STRESS_CACHE_DIR:-${PROJECT_DIR}/data/processed_stress_peeq_150}"
export INSPECT_DIR="${INSPECT_DIR:-${PROJECT_DIR}/data/inspect_report}"
export RUNS_DIR="${RUNS_DIR:-${PROJECT_DIR}/runs}"
export PRED_DIR="${PRED_DIR:-${PROJECT_DIR}/predictions}"
export LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "$LOG_DIR" "$RUNS_DIR" "$PRED_DIR" "${PROJECT_DIR}/data"

VENV_TAR_DEFAULT="/data/run01/scxk573/lsz/codes/physi-ai/venv.tar"

prepare_springback_env() {
    local expected_gpus="$1"
    local tag="$2"
    local omp_threads="${3:-1}"
    local venv_tar="${VENV_TAR_OVERRIDE:-$VENV_TAR_DEFAULT}"
    local job_tmp="/dev/shm/${USER}_${tag}_${SLURM_JOB_ID:-$$}"
    local venv_dir="$job_tmp/.venv"
    export PYTHON_BIN=""

    cleanup_springback_env() {
        if [[ -n "${GPU_MON_PID:-}" ]]; then
            kill "$GPU_MON_PID" 2>/dev/null || true
            wait "$GPU_MON_PID" 2>/dev/null || true
        fi
        if [[ "${KEEP_JOB_TMP:-0}" != "1" ]]; then rm -rf "$job_tmp" || true; fi
    }
    trap cleanup_springback_env EXIT

    rm -rf "$job_tmp"
    mkdir -p "$job_tmp" "$job_tmp/tmp" "$job_tmp/matplotlib"

    if [[ -f "$venv_tar" ]]; then
        echo "[ENV] unpacking Python environment: $venv_tar"
        tar -xf "$venv_tar" -C "$job_tmp"
        PYTHON_BIN="$venv_dir/bin/python"
    else
        echo "[ENV] WARNING: $venv_tar not found; falling back to current python"
        PYTHON_BIN="$(command -v python || true)"
    fi
    [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { echo "ERROR: usable Python not found"; exit 2; }

    export PYTHONUNBUFFERED=1
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export OMP_NUM_THREADS="$omp_threads"
    export MKL_NUM_THREADS="$omp_threads"
    export TMPDIR="$job_tmp/tmp"
    export MPLCONFIGDIR="$job_tmp/matplotlib"

    "$PYTHON_BIN" - "$expected_gpus" <<'PY'
import sys, torch
n=int(sys.argv[1])
print('python         :',sys.executable)
print('torch          :',torch.__version__)
print('cuda available :',torch.cuda.is_available())
print('cuda devices   :',torch.cuda.device_count())
if n > 0:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if torch.cuda.device_count() != n:
        raise RuntimeError('Expected {} visible GPUs, got {}'.format(n, torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print('GPU {}: {} | {:.2f} GiB'.format(i,torch.cuda.get_device_name(i),p.total_memory/1024**3))
PY
}

start_gpu_monitor() {
    local logfile="$1"
    mkdir -p "$(dirname "$logfile")"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$logfile" 2>&1 &
    GPU_MON_PID=$!
}

print_hpc_paths() {
    echo "PROJECT_DIR         = $PROJECT_DIR"
    echo "SPRINGBACK_ZIP_DIR  = $SPRINGBACK_ZIP_DIR"
    echo "SPRINGBACK_PAIR_DIR = $SPRINGBACK_PAIR_DIR"
    echo "GEOM_CACHE_DIR      = $GEOM_CACHE_DIR"
    echo "STRESS_CACHE_DIR    = $STRESS_CACHE_DIR"
    echo "INSPECT_DIR         = $INSPECT_DIR"
    echo "RUNS_DIR            = $RUNS_DIR"
    echo "PRED_DIR            = $PRED_DIR"
    echo "LOG_DIR             = $LOG_DIR"
}
