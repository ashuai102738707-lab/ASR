# Project Operating Notes

This project uses a local-edit, remote-run workflow.

Local machine:
- Edit code and experiment configs here.
- Commit changes with Git.
- Push to the remote Git repository before starting a server experiment.

Remote server:
- Pull code from Git.
- Keep datasets outside the Git repository.
- Keep experiment outputs outside the Git repository.

Expected remote layout:
- Code: `/data/$USER/project`
- Datasets: `/data/$USER/datasets`
- Experiment outputs: `/data/$USER/experiments/project`

Run experiments through:
- Local launcher: `scripts/run_remote.ps1`
- Server runner: `scripts/server_run_exp.sh`

Do not commit:
- datasets
- checkpoints
- logs
- tensorboard runs
- model weights
- local secrets or server addresses
