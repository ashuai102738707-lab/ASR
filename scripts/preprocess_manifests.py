#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from asr_qwen3.syllable import segment_syllables


LANGUAGE_TO_SCRIPT = {
    "th_th": "th",
    "lo_la": "lo",
    "km_kh": "km",
    "my_mm": "my",
}


SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize manifests and add syllable annotations.")
    parser.add_argument("--manifest-dir", required=True, help="Directory containing train/validation/test JSONL.")
    parser.add_argument("--output-dir", required=True, help="Output directory for preprocessed JSONL files.")
    parser.add_argument("--sample-per-language", type=int, default=20)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess_split(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    output: list[dict[str, object]] = []
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "examples": 0,
            "chars": 0,
            "syllables": 0,
            "empty_after_normalization": 0,
        }
    )

    for row in rows:
        language = str(row.get("language", ""))
        script = LANGUAGE_TO_SCRIPT.get(language)
        text = normalize_text(str(row.get("text", "")))
        if not text:
            stats[language]["empty_after_normalization"] += 1
            continue
        syllables = [item.text for item in segment_syllables(text, script=script)]
        item = dict(row)
        item["text"] = text
        item["text_normalized"] = text
        item["syllables"] = syllables
        item["syllable_count"] = len(syllables)
        item["char_count"] = len(text)
        output.append(item)

        stats[language]["examples"] += 1
        stats[language]["chars"] += len(text)
        stats[language]["syllables"] += len(syllables)

    for language, values in stats.items():
        examples = max(values["examples"], 1)
        values["chars_per_example"] = values["chars"] / examples
        values["syllables_per_example"] = values["syllables"] / examples
        values["chars_per_syllable"] = values["chars"] / max(values["syllables"], 1)

    return output, dict(sorted(stats.items()))


def collect_samples(rows: list[dict[str, object]], sample_per_language: int) -> dict[str, list[dict[str, object]]]:
    samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        language = str(row.get("language", ""))
        if len(samples[language]) >= sample_per_language:
            continue
        samples[language].append(
            {
                "id": row.get("id"),
                "text": row.get("text_normalized"),
                "syllables": row.get("syllables"),
            }
        )
    return dict(sorted(samples.items()))


def main() -> None:
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {"splits": {}, "samples": {}}
    for split in ("train", "validation", "test"):
        rows = read_jsonl(manifest_dir / f"{split}.jsonl")
        processed, stats = preprocess_split(rows)
        write_jsonl(output_dir / f"{split}.jsonl", processed)
        report["splits"][split] = {
            "input_examples": len(rows),
            "output_examples": len(processed),
            "languages": stats,
        }
        if split == "train":
            report["samples"] = collect_samples(processed, args.sample_per_language)

    report_path = output_dir / "preprocess_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote preprocessed manifests to {output_dir}")
    print(f"Wrote preprocess report to {report_path}")


if __name__ == "__main__":
    main()
