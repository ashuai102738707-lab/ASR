#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer, get_linear_schedule_with_warmup

from asr_qwen3.config import load_config
from asr_qwen3.data import SpeechTextCollator, SpeechTextDataset
from asr_qwen3.modeling import SpeechToQwen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train speech encoder + Qwen3 decoder ASR.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--data-root", required=True, help="FLEURS data root. Kept for runner compatibility.")
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Any, device: torch.device) -> dict[str, Any]:
    return {
        "input_values": batch.input_values.to(device),
        "input_attention_mask": batch.input_attention_mask.to(device),
        "decoder_input_ids": batch.decoder_input_ids.to(device),
        "decoder_attention_mask": batch.decoder_attention_mask.to(device),
        "labels": batch.labels.to(device),
    }


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def maybe_apply_lora(model: SpeechToQwen, train_cfg: dict[str, Any]) -> None:
    lora_cfg = train_cfg.get("lora", {})
    if not isinstance(lora_cfg, dict) or not lora_cfg.get("enabled", False):
        return
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required when train.lora.enabled=true") from exc

    targets = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model.llm = get_peft_model(model.llm, peft_config)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = dict(config.get("train", {}))
    model_cfg = dict(config.get("model", {}))
    data_cfg = dict(config.get("data", {}))

    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(train_cfg.get("fp16", True)) and device.type == "cuda"
    dtype = torch.float16 if use_fp16 else None

    llm_name = str(model_cfg.get("llm_name", "Qwen/Qwen3-0.6B"))
    speech_encoder_name = str(model_cfg.get("speech_encoder_name", "facebook/wav2vec2-xls-r-300m"))
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    feature_extractor = AutoFeatureExtractor.from_pretrained(speech_encoder_name, trust_remote_code=True)

    manifest_dir = output_dir / "manifests"
    train_manifest = Path(str(data_cfg.get("train_manifest", manifest_dir / "train.jsonl")))
    languages = config.get("languages")
    if not isinstance(languages, list):
        languages = None

    dataset = SpeechTextDataset(
        manifest_path=train_manifest,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        languages=[str(item) for item in languages] if languages else None,
        max_duration_seconds=float(data_cfg["max_duration_seconds"])
        if "max_duration_seconds" in data_cfg
        else None,
        max_text_length=int(data_cfg.get("max_text_length", 256)),
        sample_limit=int(data_cfg["sample_limit"]) if data_cfg.get("sample_limit") else None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No training examples found in {train_manifest}")

    collator = SpeechTextCollator(pad_token_id=int(tokenizer.pad_token_id))
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )

    model = SpeechToQwen(
        speech_encoder_name=speech_encoder_name,
        llm_name=llm_name,
        prefix_length=int(model_cfg.get("prefix_length", 64)),
        freeze_speech_encoder=bool(model_cfg.get("freeze_speech_encoder", True)),
        freeze_llm=bool(model_cfg.get("freeze_llm", True)),
        torch_dtype=dtype,
    )
    maybe_apply_lora(model, train_cfg)
    if bool(train_cfg.get("gradient_checkpointing", False)):
        model.llm.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    parameter_report = count_parameters(model)
    (output_dir / "parameter_report.json").write_text(
        json.dumps(parameter_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(parameter_report, indent=2))

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(train_cfg.get("epochs", 1))
    max_steps = int(train_cfg.get("max_steps", 0))
    total_steps = max_steps or epochs * len(loader)
    warmup_steps = int(train_cfg.get("warmup_steps", max(1, math.floor(total_steps * 0.03))))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    log_every = int(train_cfg.get("log_every", 10))
    save_every = int(train_cfg.get("save_every", 0))
    global_step = 0
    running_loss = 0.0

    for epoch in range(epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(progress):
            tensors = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = model(**tensors)
                if output.loss is None:
                    raise RuntimeError("Model did not return a loss.")
                loss = output.loss / grad_accum

            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * grad_accum

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = running_loss / log_every
                    progress.set_postfix(loss=f"{avg_loss:.4f}", step=global_step)
                    running_loss = 0.0

                if save_every and global_step % save_every == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.projector.state_dict(), checkpoint_dir / "projector.pt")

                if max_steps and global_step >= max_steps:
                    break
        if max_steps and global_step >= max_steps:
            break

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), final_dir / "projector.pt")
    tokenizer.save_pretrained(final_dir / "tokenizer")
    print(f"Finished training at step {global_step}. Saved projector to {final_dir}")


if __name__ == "__main__":
    main()
