#!/bin/bash
#SBATCH -J ysy_sb_cascade
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -o ./logs/%x-%j.out
#SBATCH -e ./logs/%x-%j.err
set -euo pipefail

PROJECT_DIR="$(readlink -f "${SLURM_SUBMIT_DIR:?submit from MGN_springback_onestep_150_singlemat}")"
source "$PROJECT_DIR/hpc_common.sh"
prepare_springback_env 1 sb_cascade_exp04 6
start_gpu_monitor "$PROJECT_DIR/logs/gpu_cascade_${SLURM_JOB_ID}.csv"
cd "$PROJECT_DIR"

STRUCT_ROOT="/data/home/scxk573/run/lsz/ysy/physicsnemo/examples/structural_mechanics"
BEND_PROJECT="${STRUCT_ROOT}/FINAL_BENDING_EXPERIMENTS"
BEND_RAW_DIR="/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_multifield_150_npz_compact"
BEND_PRED_DIR="${BEND_PROJECT}/rollout/exp04_mgnt_h128_x10_local_anaresid_2step/test_pred_npz"
AW_CKPT="${PROJECT_DIR}/runs/AW_SpringMGN_150/best_model.pt"
OUT="${PROJECT_DIR}/predictions/AW_SpringMGN_150_cascade_exp04_test"
ORACLE="${PROJECT_DIR}/predictions/AW_SpringMGN_150_test"

# Canonicalize /data/home aliases on compute nodes where possible.
for v in BEND_PROJECT BEND_RAW_DIR BEND_PRED_DIR AW_CKPT ORACLE; do
    p="${!v}"
    if [[ -e "$p" ]]; then printf -v "$v" '%s' "$(readlink -f "$p")"; fi
done

[[ -f "$AW_CKPT" ]] || { echo "ERROR: missing AW checkpoint: $AW_CKPT"; exit 2; }
[[ -d "$BEND_RAW_DIR" ]] || { echo "ERROR: missing bending raw dir: $BEND_RAW_DIR"; exit 3; }
[[ -d "$BEND_PRED_DIR" ]] || { echo "ERROR: missing bending prediction dir: $BEND_PRED_DIR"; exit 4; }
[[ -d "$SPRINGBACK_PAIR_DIR" ]] || { echo "ERROR: missing springback pair dir: $SPRINGBACK_PAIR_DIR"; exit 5; }

N_BEND="$(find "$BEND_PRED_DIR" -maxdepth 1 -type f -name 'sample_*_rollout.npz' | wc -l)"
echo "bending rollout NPZ count = $N_BEND"
[[ "$N_BEND" -eq 15 ]] || {
    echo "ERROR: expected 15 exp04 Test15 bending rollout NPZ files."
    echo "Run the optional exp04 4090 inference script first."
    exit 6
}

rm -rf "$OUT"
mkdir -p "$OUT"

echo "================================================================================"
echo "CASCADE PREFLIGHT"
echo "Bending : exp04 Local-ANARESID X10 2-step"
echo "AW ckpt : $AW_CKPT"
echo "Output  : $OUT"
echo "================================================================================"

"$PYTHON_BIN" -u predict_cascade_from_bending.py \
  --ckpt "$AW_CKPT" \
  --split test \
  --springback_data_dir "$SPRINGBACK_PAIR_DIR" \
  --bending_raw_dir "$BEND_RAW_DIR" \
  --bending_pred_dir "$BEND_PRED_DIR" \
  --out_dir "$OUT" \
  --expected_graphs 540 \
  --expected_samples 15 \
  --expected_angles_per_sample 36 \
  --preflight_only

echo "================================================================================"
echo "CASCADE TEST540"
echo "================================================================================"

"$PYTHON_BIN" -u predict_cascade_from_bending.py \
  --ckpt "$AW_CKPT" \
  --split test \
  --springback_data_dir "$SPRINGBACK_PAIR_DIR" \
  --bending_raw_dir "$BEND_RAW_DIR" \
  --bending_pred_dir "$BEND_PRED_DIR" \
  --out_dir "$OUT" \
  --expected_graphs 540 \
  --expected_samples 15 \
  --expected_angles_per_sample 36 \
  --save_case_npz

"$PYTHON_BIN" -u calc_springback_angle_errors_npz.py \
  --case_dir "$OUT/cases_npz" \
  --out_csv "$OUT/angle_metrics_centerline.csv" \
  --rings 20

if [[ -d "$ORACLE" && -f "$ORACLE/metrics_test.csv" ]]; then
  "$PYTHON_BIN" -u summarize_cascade.py --cascade_dir "$OUT" --oracle_dir "$ORACLE"
else
  echo "WARNING: Oracle result directory not found; skipping direct Oracle/Cascade comparison: $ORACLE"
fi

echo "================================================================================"
echo "CASCADE COMPLETE"
echo "metrics : $OUT/metrics_cascade_test.csv"
echo "angles  : $OUT/angle_metrics_centerline.csv"
echo "summary : $OUT/cascade_summary.json"
echo "compare : $OUT/oracle_vs_cascade_summary.json"
echo "================================================================================"
