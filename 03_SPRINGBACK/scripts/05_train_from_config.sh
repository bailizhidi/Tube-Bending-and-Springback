#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/hpc_common.sh"
CFG="${1:?Usage: ./scripts/05_train_from_config.sh conf/config_AW_150.yaml [output_name]}"
NAME="${2:-$(basename "$CFG" .yaml)}"
case "$(basename "$CFG")" in
    config_B_stress_peeq_dataonly_150.yaml) CACHE="$STRESS_CACHE_DIR" ;;
    *) CACHE="$GEOM_CACHE_DIR" ;;
esac
cd "$PROJECT_DIR"
PY="${PYTHON_BIN:-$(command -v python)}"
NGPU="${NGPU:-1}"
OUT="$RUNS_DIR/$NAME"
extra=()
if [[ -n "${RESUME:-}" ]]; then extra+=(--resume "$RESUME"); fi
if [[ "$NGPU" -eq 1 ]]; then
    "$PY" train_ddp_cached.py --config "$CFG" --processed_dir "$CACHE" --output_dir "$OUT" "${extra[@]}"
else
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" train_ddp_cached.py \
      --config "$CFG" --processed_dir "$CACHE" --output_dir "$OUT" "${extra[@]}"
fi
