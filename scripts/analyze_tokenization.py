#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from asr_qwen3.manifest import load_manifest, write_json
from asr_qwen3.token_fragmentation import analyze_manifest


DEFAULT_LANGUAGE_TO_SCRIPT = {
    "th_th": "th",
    "lo_la": "lo",
    "km_kh": "km",
    "my_mm": "my",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure tokenizer/syllable fragmentation.")
    parser.add_argument("--manifest", required=True, help="Prepared JSONL manifest.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--tokenizer", action="append", required=True, help="HF tokenizer name.")
    parser.add_argument("--languages", default="", help="Comma-separated language ids.")
    parser.add_argument("--max-examples-per-language", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    items = load_manifest(args.manifest, languages=languages or None)
    max_examples = args.max_examples_per_language or None
    reports = [
        analyze_manifest(
            items=items,
            tokenizer_name=tokenizer,
            language_to_script=DEFAULT_LANGUAGE_TO_SCRIPT,
            max_examples_per_language=max_examples,
        )
        for tokenizer in args.tokenizer
    ]
    write_json(Path(args.output), {"manifest": args.manifest, "reports": reports})
    print(f"Wrote tokenization report: {args.output}")


if __name__ == "__main__":
    main()
