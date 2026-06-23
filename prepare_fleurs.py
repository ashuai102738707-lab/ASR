#!/usr/bin/env python3
"""Prepare FLEURS-style ASR manifests.

The script scans language folders under a FLEURS `data/` directory, extracts
audio archives when needed, and writes JSONL manifests for train/validation/test.
It avoids moving source data and writes all generated files under `--output-dir`.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
SPLITS = ("train", "validation", "test")
TEXT_CANDIDATES = (
    "{split}.tsv",
    "{split}.csv",
    "{split}.jsonl",
    "{split}.json",
    "transcription_{split}.tsv",
    "transcription_{split}.csv",
    "metadata_{split}.csv",
)
TEXT_KEYS = ("transcription", "sentence", "text", "raw_transcription")
ID_KEYS = ("id", "file_name", "path", "audio", "audio_path", "filename")


@dataclass(frozen=True)
class Example:
    item_id: str
    language: str
    split: str
    audio_path: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FLEURS ASR manifests.")
    parser.add_argument("--config", default="", help="Optional experiment config.")
    parser.add_argument("--data-root", required=True, help="FLEURS data directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--languages",
        default="",
        help="Comma-separated language ids. Defaults to every directory.",
    )
    parser.add_argument(
        "--extract-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract split audio tarballs into the output directory.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, object]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}


def config_languages(config: dict[str, object]) -> str:
    languages = config.get("languages")
    if isinstance(languages, list):
        return ",".join(str(item) for item in languages)
    if isinstance(languages, str):
        return languages
    return ""


def config_extract_audio(config: dict[str, object], default: bool) -> bool:
    prepare = config.get("prepare")
    if isinstance(prepare, dict) and isinstance(prepare.get("extract_audio"), bool):
        return bool(prepare["extract_audio"])
    return default


def read_rows(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            for key in ("data", "examples", "rows"):
                if isinstance(loaded.get(key), list):
                    return loaded[key]
        raise ValueError(f"Unsupported JSON structure: {path}")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported text file type: {path}")


def find_text_file(language_dir: Path, split: str) -> Path | None:
    for pattern in TEXT_CANDIDATES:
        candidate = language_dir / pattern.format(split=split)
        if candidate.exists():
            return candidate
    for candidate in sorted(language_dir.rglob("*")):
        if not candidate.is_file():
            continue
        lower_name = candidate.name.lower()
        if split in lower_name and candidate.suffix.lower() in {".csv", ".tsv", ".jsonl", ".json"}:
            return candidate
    return None


def extract_tarball(tarball: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"
    if marker.exists():
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as archive:
        archive.extractall(extract_dir)
    marker.write_text(str(tarball), encoding="utf-8")


def find_audio_root(language_dir: Path, split: str, output_dir: Path, extract_audio: bool) -> Path:
    tarball = language_dir / "audio" / f"{split}.tar.gz"
    if tarball.exists() and extract_audio:
        extract_dir = output_dir / "audio" / language_dir.name / split
        extract_tarball(tarball, extract_dir)
        return extract_dir

    candidates = [
        language_dir / "audio" / split,
        language_dir / split,
        language_dir / "audio",
        language_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and any_audio_files(candidate):
            return candidate
    return language_dir


def any_audio_files(path: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS for p in path.rglob("*"))


def index_audio_files(audio_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(audio_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        keys = {
            path.name,
            path.stem,
            str(path),
            str(path.as_posix()),
        }
        for key in keys:
            index[key] = path
    return index


def first_present(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def resolve_audio(row: dict[str, str], audio_index: dict[str, Path]) -> Path | None:
    row_id = first_present(row, ID_KEYS)
    candidates = [row_id, Path(row_id).name, Path(row_id).stem]
    for key, value in row.items():
        if value and Path(str(value)).suffix.lower() in AUDIO_EXTENSIONS:
            candidates.extend([str(value), Path(str(value)).name, Path(str(value)).stem])
    for candidate in candidates:
        if candidate in audio_index:
            return audio_index[candidate]
    return None


def build_examples(
    language_dir: Path,
    split: str,
    output_dir: Path,
    extract_audio: bool,
) -> tuple[list[Example], dict[str, int | str | None]]:
    text_file = find_text_file(language_dir, split)
    audio_root = find_audio_root(language_dir, split, output_dir, extract_audio)
    audio_index = index_audio_files(audio_root)

    if text_file is None:
        return [], {
            "language": language_dir.name,
            "split": split,
            "text_file": None,
            "audio_root": str(audio_root),
            "rows": 0,
            "matched": 0,
            "missing_audio": len(audio_index),
        }

    rows = read_rows(text_file)
    examples: list[Example] = []
    missing_audio = 0
    missing_text = 0

    for row in rows:
        text = first_present(row, TEXT_KEYS)
        if not text:
            missing_text += 1
            continue
        audio_path = resolve_audio(row, audio_index)
        if audio_path is None:
            missing_audio += 1
            continue
        item_id = first_present(row, ID_KEYS) or audio_path.stem
        examples.append(
            Example(
                item_id=item_id,
                language=language_dir.name,
                split=split,
                audio_path=str(audio_path),
                text=text,
            )
        )

    return examples, {
        "language": language_dir.name,
        "split": split,
        "text_file": str(text_file),
        "audio_root": str(audio_root),
        "rows": len(rows),
        "matched": len(examples),
        "missing_audio": missing_audio,
        "missing_text": missing_text,
    }


def write_jsonl(path: Path, examples: list[Example]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "id": example.item_id,
                        "language": example.language,
                        "split": example.split,
                        "audio_path": example.audio_path,
                        "text": example.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    languages = args.languages or config_languages(config)
    extract_audio = config_extract_audio(config, args.extract_audio)
    requested = {item.strip() for item in languages.split(",") if item.strip()}
    language_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if requested:
        language_dirs = [p for p in language_dirs if p.name in requested]

    if not language_dirs:
        raise RuntimeError(f"No language directories found under {data_root}")

    summary: list[dict[str, int | str | None]] = []
    all_examples_by_split: dict[str, list[Example]] = {split: [] for split in SPLITS}

    for language_dir in language_dirs:
        for split in SPLITS:
            examples, stats = build_examples(
                language_dir=language_dir,
                split=split,
                output_dir=output_dir,
                extract_audio=extract_audio,
            )
            summary.append(stats)
            all_examples_by_split[split].extend(examples)

    for split, examples in all_examples_by_split.items():
        write_jsonl(manifest_dir / f"{split}.jsonl", examples)

    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote manifests to {manifest_dir}")
    print(f"Wrote summary to {summary_path}")
    for split, examples in all_examples_by_split.items():
        print(f"{split}: {len(examples)} examples")


if __name__ == "__main__":
    main()
