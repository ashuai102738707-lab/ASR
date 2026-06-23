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

WORKSPACE_ROOT="$(cd "$(dirname "$REMOTE_EXPERIMENT_ROOT")/../.." && pwd)"
if [[ "$REMOTE_EXPERIMENT_ROOT" == /datasets/xhm_files/* ]]; then
  WORKSPACE_ROOT="/datasets/xhm_files"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

EXP_NAME="$(basename "$CONFIG")"
EXP_NAME="${EXP_NAME%.*}"
RUN_DIR="${REMOTE_EXPERIMENT_ROOT}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

cp "$CONFIG" "$RUN_DIR/config.yaml"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$REMOTE_CONDA_ENV" ]]; then
  echo "Using active conda environment: $CONDA_DEFAULT_ENV"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$REMOTE_CONDA_ENV"
else
  echo "conda is not available on PATH and active env is '${CONDA_DEFAULT_ENV:-none}', expected '$REMOTE_CONDA_ENV'" >&2
  exit 1
fi

echo "Run directory: $RUN_DIR"
echo "Config: $CONFIG"
echo "Data root: $REMOTE_DATA_ROOT"
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

{
  echo "Python: $(python --version 2>&1)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi:"
    nvidia-smi
  else
    echo "nvidia-smi: not found"
  fi
  python - <<'PY'
import json
import platform

report = {
    "python": platform.python_version(),
    "platform": platform.platform(),
}

try:
    import torch

    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    report["gpu_count"] = torch.cuda.device_count()
    report["gpus"] = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]
except Exception as exc:
    report["torch_error"] = repr(exc)

print(json.dumps(report, ensure_ascii=False, indent=2))
PY
} 2>&1 | tee "$RUN_DIR/env_report.log"

if [[ -f prepare_fleurs.py ]]; then
  python prepare_fleurs.py \
    --config "$CONFIG" \
    --data-root "$REMOTE_DATA_ROOT" \
    --output-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/prepare.log"
else
  echo "No prepare_fleurs.py found. Skipping data preparation." | tee "$RUN_DIR/prepare.log"
fi

python - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
summary_path = run_dir / "prepare_summary.json"
if not summary_path.exists():
    raise SystemExit("prepare_summary.json was not produced")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
matched = sum(int(item.get("matched", 0) or 0) for item in summary)
if matched == 0:
    raise SystemExit(
        "No manifest examples were matched. Check REMOTE_DATA_ROOT and FLEURS TSV/audio layout."
    )
print(f"Matched manifest examples: {matched}")
PY

if [[ -f scripts/preprocess_manifests.py ]]; then
  python scripts/preprocess_manifests.py \
    --manifest-dir "$RUN_DIR/manifests" \
    --output-dir "$RUN_DIR/preprocessed" \
    2>&1 | tee "$RUN_DIR/preprocess.log"
fi

if [[ -f scripts/analyze_tokenization.py ]]; then
  ANALYSIS_ENABLED="$(python - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(str((cfg.get("analysis") or {}).get("enabled", False)).lower())
PY
)"
  if [[ "$ANALYSIS_ENABLED" == "true" ]]; then
    python - "$CONFIG" "$RUN_DIR" <<'PY' | bash 2>&1 | tee "$RUN_DIR/tokenization.log"
import shlex
import sys
import yaml
from pathlib import Path

config_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
analysis = cfg.get("analysis") or {}
split = analysis.get("split", "train")
preprocessed_manifest = run_dir / "preprocessed" / f"{split}.jsonl"
manifest = preprocessed_manifest if preprocessed_manifest.exists() else run_dir / "manifests" / f"{split}.jsonl"
output = run_dir / "tokenization_fragmentation.json"
cmd = [
    "python",
    "scripts/analyze_tokenization.py",
    "--manifest",
    str(manifest),
    "--output",
    str(output),
]
for tokenizer in analysis.get("tokenizers", []):
    cmd.extend(["--tokenizer", str(tokenizer)])
languages = cfg.get("languages") or []
if languages:
    cmd.extend(["--languages", ",".join(str(item) for item in languages)])
max_examples = int(analysis.get("max_examples_per_language", 0) or 0)
if max_examples:
    cmd.extend(["--max-examples-per-language", str(max_examples)])
print(" ".join(shlex.quote(part) for part in cmd))
PY
  else
    echo "Tokenization analysis disabled by config." | tee "$RUN_DIR/tokenization.log"
  fi
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
