from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .manifest import ManifestItem, load_manifest


@dataclass
class Batch:
    input_values: torch.Tensor
    input_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: torch.Tensor
    languages: list[str]
    texts: list[str]


class SpeechTextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: Any,
        feature_extractor: Any,
        languages: list[str] | None = None,
        max_duration_seconds: float | None = None,
        max_text_length: int = 256,
        sample_limit: int | None = None,
    ) -> None:
        self.items = load_manifest(manifest_path, languages=languages)
        if sample_limit is not None:
            self.items = self.items[:sample_limit]
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_duration_seconds = max_duration_seconds
        self.max_text_length = max_text_length
        self.target_sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        waveform, sampling_rate = torchaudio.load(item.audio_path)
        waveform = waveform.mean(dim=0)
        if sampling_rate != self.target_sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sampling_rate,
                new_freq=self.target_sampling_rate,
            )
        if self.max_duration_seconds:
            max_samples = int(self.max_duration_seconds * self.target_sampling_rate)
            waveform = waveform[:max_samples]
        features = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.target_sampling_rate,
            return_attention_mask=True,
        )
        tokenized = self.tokenizer(
            item.text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_text_length,
        )
        return {
            "input_values": torch.tensor(features["input_values"][0], dtype=torch.float32),
            "input_attention_mask": torch.tensor(
                features.get("attention_mask", [[1] * len(features["input_values"][0])])[0],
                dtype=torch.long,
            ),
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "language": item.language,
            "text": item.text,
        }


class SpeechTextCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> Batch:
        input_values = pad_sequence(
            [example["input_values"] for example in examples],
            batch_first=True,
            padding_value=0.0,
        )
        input_attention_mask = pad_sequence(
            [example["input_attention_mask"] for example in examples],
            batch_first=True,
            padding_value=0,
        )
        decoder_input_ids = pad_sequence(
            [example["input_ids"] for example in examples],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        decoder_attention_mask = (decoder_input_ids != self.pad_token_id).long()
        labels = decoder_input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        return Batch(
            input_values=input_values,
            input_attention_mask=input_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            languages=[str(example["language"]) for example in examples],
            texts=[str(example["text"]) for example in examples],
        )
