#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/hpc_common.sh"
print_hpc_paths
mkdir -p "$SPRINGBACK_ZIP_DIR" "$LOG_DIR"
echo "=== upload directory contents ==="
ls -lh "$SPRINGBACK_ZIP_DIR" || true
