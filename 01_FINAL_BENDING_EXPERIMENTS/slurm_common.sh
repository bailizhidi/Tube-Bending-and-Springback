#!/bin/bash
set -euo pipefail

prepare_anaresid_env() {
    local expected_gpus="$1"
    local tag="$2"
    PROJECT="${PROJECT_OVERRIDE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    VENV_TAR="${VENV_TAR_OVERRIDE:-/data/run01/scxk573/lsz/codes/physi-ai/venv.tar}"
    JOB_TMP="/dev/shm/${USER}_${tag}_${SLURM_JOB_ID:-$$}"
    VENV_DIR="$JOB_TMP/.venv"
    PYTHON_BIN="$VENV_DIR/bin/python"
    cd "$PROJECT"

    cleanup_anaresid() {
        if [[ -n "${GPU_MON_PID:-}" ]]; then
            kill "$GPU_MON_PID" 2>/dev/null || true
            wait "$GPU_MON_PID" 2>/dev/null || true
        fi
        rm -rf "$JOB_TMP" || true
    }
    trap cleanup_anaresid EXIT

    rm -rf "$JOB_TMP"
    mkdir -p "$JOB_TMP"
    tar -xf "$VENV_TAR" -C "$JOB_TMP"
    [[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing $PYTHON_BIN"; exit 1; }
    export VIRTUAL_ENV="$VENV_DIR"
    export PATH="$VENV_DIR/bin:$PATH"
    export PYTHONUNBUFFERED=1
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export OMP_NUM_THREADS=6
    export MPLCONFIGDIR="$JOB_TMP/matplotlib"
    export TMPDIR="$JOB_TMP/tmp"
    mkdir -p "$MPLCONFIGDIR" "$TMPDIR" logs

    "$PYTHON_BIN" - "$expected_gpus" <<'PY'
import sys, torch
n=int(sys.argv[1])
print('torch          :',torch.__version__)
print('cuda available :',torch.cuda.is_available())
print('cuda devices   :',torch.cuda.device_count())
if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
if torch.cuda.device_count()!=n: raise RuntimeError(f'Expected {n} visible GPUs, got {torch.cuda.device_count()}')
for i in range(n):
    p=torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {torch.cuda.get_device_name(i)} | {p.total_memory/1024**3:.2f} GiB')
PY
}

start_gpu_monitor() {
    local logfile="$1"
    mkdir -p "$(dirname "$logfile")"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv -l 1 > "$logfile" 2>&1 &
    GPU_MON_PID=$!
}
