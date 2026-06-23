from __future__ import annotations

import os
from pathlib import Path


def resolve_local_snapshot(model_name_or_path: str, local_files_only: bool = True) -> str:
    path = Path(model_name_or_path)
    if path.exists() or "/" not in model_name_or_path or not local_files_only:
        return model_name_or_path

    cache_root = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    )
    repo_dir = cache_root / f"models--{model_name_or_path.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name_or_path

    ref_path = repo_dir / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot = snapshots_dir / revision
        if snapshot.exists():
            return str(snapshot)

    snapshots = sorted(
        [item for item in snapshots_dir.iterdir() if item.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if snapshots:
        return str(snapshots[0])
    return model_name_or_path
