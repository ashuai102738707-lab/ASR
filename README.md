# Qwen3-ASR for Syllable Fragmentation Experiments

This repository implements the remote-run workflow for the Qwen3-based ASR
experiments on Thai, Lao, Khmer, and Burmese FLEURS subsets.

## Research Route

The paper route is:

1. Measure tokenizer/syllable fragmentation for Qwen3 and Whisper tokenizers.
2. Train a speech encoder + acoustic prefix adapter + Qwen3 decoder baseline.
3. Add syllable-aware objectives/adapters in follow-up experiments.
4. Compare CER/SyER, deletion/substitution errors, and alignment/attention
   stability against the fragmentation metrics.

The current code provides:

- FLEURS-style manifest preparation.
- Rule-based syllable segmentation for Thai/Lao/Khmer/Myanmar scripts.
- Token boundary vs syllable boundary fragmentation analysis.
- A runnable `SpeechToQwen` baseline:
  `wav2vec2/XLS-R encoder -> projected speech prefix -> Qwen3 decoder`.

## Expected Server Layout

From the current server screenshot, use:

```text
Code:        /home/speech/ASR
Data:        /home/speech/fleurs_subset
Experiments: /home/speech/experiments/qwen3_asr
SSH:         speech@222.197.200.131 -p 40022
```

Adjust paths if your actual home directory differs.

## Install Dependencies on the Server

```bash
cd /home/speech/ASR
conda activate asr
pip install -r requirements.txt
```

If the environment name is not `asr`, update `.env.remote` locally or export
`REMOTE_CONDA_ENV` on the server.

## Run on the Server

Smoke test:

```bash
cd /home/speech/ASR
git pull
REMOTE_CONDA_ENV=asr \
REMOTE_DATA_ROOT=/home/speech/fleurs_subset \
REMOTE_EXPERIMENT_ROOT=/home/speech/experiments/qwen3_asr \
bash scripts/server_run_exp.sh configs/exp001.yaml
```

Full Qwen3-0.6B run:

```bash
REMOTE_CONDA_ENV=asr \
REMOTE_DATA_ROOT=/home/speech/fleurs_subset \
REMOTE_EXPERIMENT_ROOT=/home/speech/experiments/qwen3_asr \
bash scripts/server_run_exp.sh configs/qwen3_0p6b_full.yaml
```

Qwen3-1.7B two-language ablation:

```bash
REMOTE_CONDA_ENV=asr \
REMOTE_DATA_ROOT=/home/speech/fleurs_subset \
REMOTE_EXPERIMENT_ROOT=/home/speech/experiments/qwen3_asr \
bash scripts/server_run_exp.sh configs/qwen3_1p7b_ablation.yaml
```

## Run from Windows

Copy the remote config:

```powershell
Copy-Item configs\remote.example.env .env.remote
```

Then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_remote.ps1 -Config configs/exp001.yaml
```

If you already have the SSH alias `gpu2` configured:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_remote.ps1 -Config configs/exp001.yaml -RemoteAlias gpu2
```

## Storage Audit

Before deleting anything on the server, run:

```bash
bash /home/speech/ASR/scripts/server_audit_storage.sh /home/speech
```

This script only reports directory sizes, cache folders, and large checkpoint
files. It does not delete files.

Likely review candidates from the screenshot:

```text
fleurs_test_processed
khmer_checkpoints_8b
km_datasets_fleurs
km_test_wavs
omni2_wav_data
LLaMA-Omni2
```

Do not delete `fleurs_subset`, `ASR`, or active experiment/checkpoint folders
unless you have confirmed they are duplicated or obsolete.
