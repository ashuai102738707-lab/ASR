from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ManifestItem:
    item_id: str
    language: str
    split: str
    audio_path: str
    text: str


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows: list[dict[str, object]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def load_manifest(path: str | Path, languages: Iterable[str] | None = None) -> list[ManifestItem]:
    requested = set(languages or [])
    items: list[ManifestItem] = []
    for row in load_jsonl(path):
        language = str(row.get("language", ""))
        if requested and language not in requested:
            continue
        items.append(
            ManifestItem(
                item_id=str(row.get("id", row.get("item_id", ""))),
                language=language,
                split=str(row.get("split", "")),
                audio_path=str(row.get("audio_path", "")),
                text=str(row.get("text", "")),
            )
        )
    return items


def write_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
