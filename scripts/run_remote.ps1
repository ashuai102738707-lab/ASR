param(
    [string]$Config = "configs/exp001.yaml",
    [string]$EnvFile = ".env.remote",
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Missing $Path. Copy configs\remote.example.env to $Path and fill in your server settings."
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) {
            return
        }

        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) {
            throw "Invalid env line in $Path`: $line"
        }

        [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
    }
}

function Require-Env {
    param([string]$Name)

    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing required setting: $Name"
    }
    return $value
}

Import-DotEnv -Path $EnvFile

$remoteHost = Require-Env "REMOTE_HOST"
$remoteProjectDir = Require-Env "REMOTE_PROJECT_DIR"
$remoteCondaEnv = Require-Env "REMOTE_CONDA_ENV"
$remoteDataRoot = Require-Env "REMOTE_DATA_ROOT"
$remoteExperimentRoot = Require-Env "REMOTE_EXPERIMENT_ROOT"

if (-not (Test-Path $Config)) {
    throw "Config file not found: $Config"
}

git rev-parse --is-inside-work-tree | Out-Null

if (-not $NoPush) {
    git status --short
    git push
}

$remoteCommand = @"
set -euo pipefail
cd '$remoteProjectDir'
git pull
REMOTE_CONDA_ENV='$remoteCondaEnv' REMOTE_DATA_ROOT='$remoteDataRoot' REMOTE_EXPERIMENT_ROOT='$remoteExperimentRoot' bash scripts/server_run_exp.sh '$Config'
"@

ssh $remoteHost $remoteCommand
