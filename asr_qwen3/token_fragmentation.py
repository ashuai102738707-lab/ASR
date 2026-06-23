from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer

from .manifest import ManifestItem
from .syllable import boundaries_from_spans, segment_syllables


@dataclass
class FragmentationTotals:
    examples: int = 0
    chars: int = 0
    syllables: int = 0
    tokens: int = 0
    token_boundaries: int = 0
    token_boundaries_inside_syllable: int = 0
    missed_syllable_boundaries: int = 0

    def as_metrics(self) -> dict[str, float | int]:
        return {
            "examples": self.examples,
            "chars": self.chars,
            "syllables": self.syllables,
            "tokens": self.tokens,
            "tokens_per_syllable": self.tokens / max(self.syllables, 1),
            "token_boundary_inside_syllable_ratio": self.token_boundaries_inside_syllable
            / max(self.token_boundaries, 1),
            "missed_syllable_boundary_ratio": self.missed_syllable_boundaries
            / max(self.syllables - self.examples, 1),
        }


def token_offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError(
            f"Tokenizer {tokenizer.__class__.__name__} does not expose offset mappings."
        )
    return [(int(start), int(end)) for start, end in offsets if int(end) > int(start)]


def score_text(tokenizer: Any, text: str, script: str | None = None) -> FragmentationTotals:
    syllables = segment_syllables(text, script=script)
    syllable_boundaries = boundaries_from_spans(syllables)
    offsets = token_offsets(tokenizer, text)
    token_boundaries = {end for _, end in offsets[:-1]}

    totals = FragmentationTotals(
        examples=1,
        chars=len(text),
        syllables=len(syllables),
        tokens=len(offsets),
        token_boundaries=len(token_boundaries),
        token_boundaries_inside_syllable=len(token_boundaries - syllable_boundaries),
        missed_syllable_boundaries=len(syllable_boundaries - token_boundaries),
    )
    return totals


def add_totals(left: FragmentationTotals, right: FragmentationTotals) -> FragmentationTotals:
    return FragmentationTotals(
        examples=left.examples + right.examples,
        chars=left.chars + right.chars,
        syllables=left.syllables + right.syllables,
        tokens=left.tokens + right.tokens,
        token_boundaries=left.token_boundaries + right.token_boundaries,
        token_boundaries_inside_syllable=left.token_boundaries_inside_syllable
        + right.token_boundaries_inside_syllable,
        missed_syllable_boundaries=left.missed_syllable_boundaries
        + right.missed_syllable_boundaries,
    )


def analyze_manifest(
    items: list[ManifestItem],
    tokenizer_name: str,
    language_to_script: dict[str, str] | None = None,
    max_examples_per_language: int | None = None,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    counts: dict[str, int] = defaultdict(int)
    totals_by_lang: dict[str, FragmentationTotals] = defaultdict(FragmentationTotals)
    global_totals = FragmentationTotals()

    for item in items:
        if max_examples_per_language is not None and counts[item.language] >= max_examples_per_language:
            continue
        counts[item.language] += 1
        script = (language_to_script or {}).get(item.language)
        scored = score_text(tokenizer, item.text, script=script)
        totals_by_lang[item.language] = add_totals(totals_by_lang[item.language], scored)
        global_totals = add_totals(global_totals, scored)

    return {
        "tokenizer": tokenizer_name,
        "overall": global_totals.as_metrics(),
        "languages": {
            language: totals.as_metrics()
            for language, totals in sorted(totals_by_lang.items())
        },
    }
