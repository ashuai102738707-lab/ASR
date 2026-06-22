#!/usr/bin/env python3
"""Minimal experiment entrypoint for validating a FLEURS-style ASR dataset.

This script is intentionally conservative: it proves the remote environment,
dataset path, and output path are wired correctly before any model training is
added. Replace or extend `main` once the target ASR model and training objective
are fixed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
TEXT_EXTENSIONS = {".csv", ".json", ".jsonl", ".tsv", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FLEURS subset inputs.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--data-root", required=True, help="FLEURS data root.")
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    text = read_text(path)

    try:
        import yaml  # type: ignore
    except ImportError:
        return {"raw_config": text}

    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {"raw_config": loaded}


def scan_dataset(data_root: Path) -> dict[str, Any]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    language_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    summary: dict[str, Any] = {
        "data_root": str(data_root),
        "language_count": len(language_dirs),
        "languages": {},
    }

    for lang_dir in language_dirs:
        audio_files = []
        text_files = []

        for path in lang_dir.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in AUDIO_EXTENSIONS:
                audio_files.append(path)
            elif suffix in TEXT_EXTENSIONS:
                text_files.append(path)

        summary["languages"][lang_dir.name] = {
            "audio_files": len(audio_files),
            "text_files": len(text_files),
            "sample_audio": str(audio_files[0]) if audio_files else None,
            "sample_text": str(text_files[0]) if text_files else None,
        }

    return summary


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    dataset_summary = scan_dataset(data_root)

    run_summary = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "config": config,
        "dataset": dataset_summary,
        "environment": {
            "REMOTE_DATA_ROOT": os.environ.get("REMOTE_DATA_ROOT"),
            "REMOTE_EXPERIMENT_ROOT": os.environ.get("REMOTE_EXPERIMENT_ROOT"),
            "REMOTE_CONDA_ENV": os.environ.get("REMOTE_CONDA_ENV"),
        },
    }

    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote dataset summary: {summary_path}")
    print(f"Detected {dataset_summary['language_count']} language directories.")

    for lang, item in dataset_summary["languages"].items():
        print(
            f"{lang}: audio={item['audio_files']} text={item['text_files']}"
        )


if __name__ == "__main__":
    main()
