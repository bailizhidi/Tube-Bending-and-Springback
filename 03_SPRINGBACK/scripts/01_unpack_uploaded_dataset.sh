#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/hpc_common.sh"
mkdir -p "$SPRINGBACK_ZIP_DIR" "$(dirname "$SPRINGBACK_PAIR_DIR")"

existing=0
if [[ -d "$SPRINGBACK_PAIR_DIR" ]]; then
    existing=$(find "$SPRINGBACK_PAIR_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l)
fi
if [[ "$existing" -eq 5400 ]]; then
    echo "[UNPACK] dataset already complete: $SPRINGBACK_PAIR_DIR (5400 NPZ)"
    exit 0
fi
if [[ "$existing" -gt 0 && "${OVERWRITE_DATASET:-0}" != "1" ]]; then
    echo "ERROR: target dataset directory exists but has $existing NPZ, not 5400."
    echo "Set OVERWRITE_DATASET=1 only after checking the directory."
    exit 2
fi

if [[ -n "${ARCHIVE:-}" ]]; then
    archive="$ARCHIVE"
else
    mapfile -t candidates < <(find "$SPRINGBACK_ZIP_DIR" -maxdepth 1 -type f \( \
        -iname 'dataset_springback_pair_Base_TC4_150*.tar.gz' -o \
        -iname 'dataset_springback_pair_Base_TC4_150*.tgz' -o \
        -iname 'dataset_springback_pair_Base_TC4_150*.tar' -o \
        -iname 'dataset_springback_pair_Base_TC4_150*.zip' \
        \) | sort)
    if [[ ${#candidates[@]} -ne 1 ]]; then
        echo "ERROR: expected exactly one springback dataset archive in $SPRINGBACK_ZIP_DIR; found ${#candidates[@]}."
        printf '  %s\n' "${candidates[@]:-<none>}"
        echo "You can select one explicitly: ARCHIVE=/full/path/file.tar.gz ./scripts/01_unpack_uploaded_dataset.sh"
        exit 3
    fi
    archive="${candidates[0]}"
fi
[[ -f "$archive" ]] || { echo "ERROR: archive not found: $archive"; exit 4; }

echo "[UNPACK] archive=$archive"
tmp="$(dirname "$SPRINGBACK_PAIR_DIR")/.springback_unpack_${USER}_$$"
rm -rf "$tmp"; mkdir -p "$tmp"
trap 'rm -rf "$tmp"' EXIT
case "$archive" in
    *.tar.gz|*.tgz) tar -xzf "$archive" -C "$tmp" ;;
    *.tar) tar -xf "$archive" -C "$tmp" ;;
    *.zip) command -v unzip >/dev/null || { echo 'ERROR: unzip not installed'; exit 5; }; unzip -q "$archive" -d "$tmp" ;;
    *) echo "ERROR: unsupported archive: $archive"; exit 6 ;;
esac

found_dir="$(find "$tmp" -type d -name 'dataset_springback_pair_Base_TC4_150' -print -quit || true)"
if [[ -n "$found_dir" ]]; then
    count=$(find "$found_dir" -maxdepth 1 -type f -name '*.npz' | wc -l)
    [[ "$count" -eq 5400 ]] || { echo "ERROR: extracted named dataset folder has $count NPZ"; exit 7; }
    if [[ -e "$SPRINGBACK_PAIR_DIR" ]]; then rm -rf "$SPRINGBACK_PAIR_DIR"; fi
    mv "$found_dir" "$SPRINGBACK_PAIR_DIR"
else
    count=$(find "$tmp" -type f -name '*.npz' | wc -l)
    [[ "$count" -eq 5400 ]] || { echo "ERROR: extracted archive contains $count NPZ, expected 5400"; exit 8; }
    if [[ -e "$SPRINGBACK_PAIR_DIR" ]]; then rm -rf "$SPRINGBACK_PAIR_DIR"; fi
    mkdir -p "$SPRINGBACK_PAIR_DIR"
    find "$tmp" -type f -name '*.npz' -exec mv -t "$SPRINGBACK_PAIR_DIR" {} +
fi

final_count=$(find "$SPRINGBACK_PAIR_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l)
echo "[UNPACK] final dataset=$SPRINGBACK_PAIR_DIR"
echo "[UNPACK] NPZ count=$final_count"
[[ "$final_count" -eq 5400 ]] || { echo 'ERROR: final count is not 5400'; exit 9; }
echo "[UNPACK] PASSED"
