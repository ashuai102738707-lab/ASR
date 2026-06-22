# Remote Experiment Workflow

This repository is set up for the workflow where Codex edits code locally and experiments run on a remote server.

## 1. Configure local remote settings

Copy the example config:

```powershell
Copy-Item configs\remote.example.env .env.remote
```

Edit `.env.remote` and fill in the real server values:

```text
REMOTE_HOST=user@server
REMOTE_PROJECT_DIR=/data/user/project
REMOTE_CONDA_ENV=myenv
REMOTE_DATA_ROOT=/data/user/datasets/my_dataset
REMOTE_EXPERIMENT_ROOT=/data/user/experiments/project
```

`.env.remote` is intentionally ignored by Git.

## 2. Put files in the right place

Commit to Git:

- source code
- `configs/*.yaml`
- `scripts/*.sh`
- dependency files such as `requirements.txt` or `environment.yml`

Keep only on the server:

- datasets
- checkpoints
- logs
- tensorboard runs
- generated model outputs

## 3. Run an experiment from Windows

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_remote.ps1 -Config configs/exp001.yaml
```

By default, this script pushes local Git commits, connects to the server, pulls latest code, and runs `scripts/server_run_exp.sh`.

To skip Git push:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_remote.ps1 -Config configs/exp001.yaml -NoPush
```

## 4. Run directly on the server

```bash
cd /data/user/project
git pull
REMOTE_CONDA_ENV=myenv \
REMOTE_DATA_ROOT=/data/user/datasets/my_dataset \
REMOTE_EXPERIMENT_ROOT=/data/user/experiments/project \
bash scripts/server_run_exp.sh configs/exp001.yaml
```
