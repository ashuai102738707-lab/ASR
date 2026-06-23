from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM


@dataclass
class SpeechQwenOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


class SpeechToQwen(nn.Module):
    """Speech encoder + length adaptor + decoder-only Qwen LM.

    The speech representation is projected into Qwen's embedding space and used
    as a soft prefix. Transcript tokens are trained with causal LM loss while
    the audio-prefix positions are masked out with label=-100.
    """

    def __init__(
        self,
        speech_encoder_name: str,
        llm_name: str,
        prefix_length: int = 64,
        freeze_speech_encoder: bool = True,
        freeze_llm: bool = True,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype
        self.speech_encoder = AutoModel.from_pretrained(
            speech_encoder_name,
            **model_kwargs,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            **model_kwargs,
        )
        speech_hidden = int(getattr(self.speech_encoder.config, "hidden_size"))
        llm_hidden = int(getattr(self.llm.config, "hidden_size"))
        self.prefix_length = prefix_length
        self.pool = nn.AdaptiveAvgPool1d(prefix_length)
        self.projector = nn.Sequential(
            nn.LayerNorm(speech_hidden),
            nn.Linear(speech_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )

        if freeze_speech_encoder:
            for param in self.speech_encoder.parameters():
                param.requires_grad = False
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def encode_audio(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.speech_encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = encoded.last_hidden_state
        pooled = self.pool(hidden.transpose(1, 2)).transpose(1, 2)
        return self.projector(pooled)

    def forward(
        self,
        input_values: torch.Tensor,
        input_attention_mask: torch.Tensor | None,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SpeechQwenOutput:
        prefix_embeds = self.encode_audio(input_values, input_attention_mask)
        token_embeds = self.llm.get_input_embeddings()(decoder_input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        batch_size = decoder_input_ids.shape[0]
        prefix_mask = torch.ones(
            batch_size,
            self.prefix_length,
            dtype=decoder_attention_mask.dtype,
            device=decoder_attention_mask.device,
        )
        attention_mask = torch.cat([prefix_mask, decoder_attention_mask], dim=1)

        lm_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, self.prefix_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            lm_labels = torch.cat([prefix_labels, labels], dim=1)

        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=lm_labels,
            use_cache=False,
        )
        return SpeechQwenOutput(loss=output.loss, logits=output.logits)
