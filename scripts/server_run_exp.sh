#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/exp001.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

: "${REMOTE_CONDA_ENV:?Set REMOTE_CONDA_ENV}"
: "${REMOTE_DATA_ROOT:?Set REMOTE_DATA_ROOT}"
: "${REMOTE_EXPERIMENT_ROOT:?Set REMOTE_EXPERIMENT_ROOT}"

EXP_NAME="$(basename "$CONFIG")"
EXP_NAME="${EXP_NAME%.*}"
RUN_DIR="${REMOTE_EXPERIMENT_ROOT}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

cp "$CONFIG" "$RUN_DIR/config.yaml"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$REMOTE_CONDA_ENV"
else
  echo "conda is not available on PATH" >&2
  exit 1
fi

echo "Run directory: $RUN_DIR"
echo "Config: $CONFIG"
echo "Data root: $REMOTE_DATA_ROOT"

if [[ -f prepare_fleurs.py ]]; then
  python prepare_fleurs.py \
    --config "$CONFIG" \
    --data-root "$REMOTE_DATA_ROOT" \
    --output-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/prepare.log"
else
  echo "No prepare_fleurs.py found. Skipping data preparation." | tee "$RUN_DIR/prepare.log"
fi

if [[ -f train.py ]]; then
  python train.py \
    --config "$CONFIG" \
    --data-root "$REMOTE_DATA_ROOT" \
    --output-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/train.log"
else
  echo "No train.py found. Replace the command in scripts/server_run_exp.sh with your project's entrypoint." | tee "$RUN_DIR/train.log"
fi
