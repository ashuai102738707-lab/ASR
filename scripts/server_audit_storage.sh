#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME}"

echo "Storage audit root: $ROOT"
echo
echo "Top-level directory sizes:"
du -h --max-depth=1 "$ROOT" 2>/dev/null | sort -h || true

echo
echo "Python/cache directories that are usually safe to remove:"
find "$ROOT" -maxdepth 3 -type d \( \
  -name "__pycache__" -o \
  -name ".pytest_cache" -o \
  -name ".cache" -o \
  -name ".ipynb_checkpoints" \
\) -print 2>/dev/null || true

echo
echo "Large checkpoint/model files:"
find "$ROOT" -maxdepth 4 -type f \( \
  -name "*.pt" -o \
  -name "*.pth" -o \
  -name "*.ckpt" -o \
  -name "*.safetensors" \
\) -printf "%s %p\n" 2>/dev/null | sort -nr | head -50 | awk '{size=$1; $1=""; printf "%.2f GB %s\n", size/1024/1024/1024, $0}' || true

cat <<'EOF'

Manual cleanup guidance:
- Usually safe after review: __pycache__, .pytest_cache, failed run logs.
- Do not delete without checking: fleurs_subset, ASR, current experiments, model checkpoints used for comparison.
- Directories visible in your screenshot that look potentially obsolete but need confirmation:
  fleurs_test_processed, khmer_checkpoints_8b, km_datasets_fleurs, km_test_wavs,
  omni2_wav_data, LLaMA-Omni2.
EOF
