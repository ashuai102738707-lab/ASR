from __future__ import annotations

import unicodedata
from dataclasses import dataclass


SCRIPT_RANGES = {
    "th": ((0x0E00, 0x0E7F),),
    "lo": ((0x0E80, 0x0EFF),),
    "km": ((0x1780, 0x17FF),),
    "my": ((0x1000, 0x109F), (0xAA60, 0xAA7F), (0xA9E0, 0xA9FF)),
}

DEPENDENT_MARK_CATEGORIES = {"Mn", "Mc", "Me"}
THAI_LAO_PREPOSED_VOWELS = set("เแโใไເແໂໃໄ")
THAI_LAO_DEPENDENT_SIGNS = set(
    "ะาำิีึืฺุู็่้๊๋์ํ"
    "ະາຳິີຶື຺ຸູັົຼ່້໊໋໌ໍ"
)
JOINERS = {
    "\u1039",  # Myanmar virama/asat in stacked consonants.
    "\u17d2",  # Khmer coeng.
    "\u0ecd",  # Lao niggahita.
}


@dataclass(frozen=True)
class Syllable:
    text: str
    start: int
    end: int


def infer_script(char: str) -> str | None:
    code = ord(char)
    for script, ranges in SCRIPT_RANGES.items():
        if any(start <= code <= end for start, end in ranges):
            return script
    return None


def is_script_char(char: str, script: str | None = None) -> bool:
    if script is None:
        return infer_script(char) is not None
    code = ord(char)
    return any(start <= code <= end for start, end in SCRIPT_RANGES.get(script, ()))


def is_dependent_mark(char: str) -> bool:
    return unicodedata.category(char) in DEPENDENT_MARK_CATEGORIES


def is_thai_lao_dependent(char: str) -> bool:
    return char in THAI_LAO_DEPENDENT_SIGNS or is_dependent_mark(char)


def segment_thai_lao(text: str, script: str) -> list[Syllable]:
    spans: list[Syllable] = []
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char.isspace():
            idx += 1
            continue
        if not is_script_char(char, script):
            spans.append(Syllable(char, idx, idx + 1))
            idx += 1
            continue

        start = idx
        if char in THAI_LAO_PREPOSED_VOWELS and idx + 1 < len(text):
            idx += 1

        idx += 1
        while idx < len(text) and is_script_char(text[idx], script) and is_thai_lao_dependent(text[idx]):
            idx += 1

        # Consume a likely final consonant when it is not the onset of a following
        # vowel-bearing syllable. This is deliberately conservative and avoids
        # turning every base consonant into a separate "syllable" for Thai/Lao.
        if (
            idx < len(text)
            and is_script_char(text[idx], script)
            and text[idx] not in THAI_LAO_PREPOSED_VOWELS
            and not is_thai_lao_dependent(text[idx])
        ):
            next_char = text[idx + 1] if idx + 1 < len(text) else ""
            if not next_char or (
                next_char not in THAI_LAO_PREPOSED_VOWELS
                and not is_thai_lao_dependent(next_char)
            ):
                idx += 1
                while idx < len(text) and is_script_char(text[idx], script) and is_thai_lao_dependent(text[idx]):
                    idx += 1

        spans.append(Syllable(text[start:idx], start, idx))
    return spans


def is_nonspacing_or_joiner(char: str) -> bool:
    return is_dependent_mark(char) or char in JOINERS


def should_break(prev: str, char: str, script: str | None) -> bool:
    if not prev:
        return False
    if char.isspace() or prev.isspace():
        return True
    if unicodedata.category(char).startswith("P") or unicodedata.category(prev).startswith("P"):
        return True
    if script and is_script_char(char, script) and is_nonspacing_or_joiner(char):
        return False
    if prev in JOINERS:
        return False
    if script and is_script_char(prev, script) and is_script_char(char, script):
        # This is a conservative rule-based fallback. It preserves combining
        # stacks and starts a new syllable at the next base script character.
        return not is_nonspacing_or_joiner(char)
    return True


def segment_syllables(text: str, script: str | None = None) -> list[Syllable]:
    if not text:
        return []
    if script in {"th", "lo"}:
        return segment_thai_lao(text, script)

    spans: list[Syllable] = []
    start = 0
    prev = ""
    active_script = script

    for idx, char in enumerate(text):
        if active_script is None:
            active_script = infer_script(char)
        if idx > start and should_break(prev, char, active_script):
            piece = text[start:idx]
            if piece and not piece.isspace():
                spans.append(Syllable(piece, start, idx))
            start = idx
            if script is None:
                active_script = infer_script(char)
        prev = char

    piece = text[start:]
    if piece and not piece.isspace():
        spans.append(Syllable(piece, start, len(text)))
    return spans


def boundaries_from_spans(spans: list[Syllable]) -> set[int]:
    return {span.end for span in spans[:-1]}
