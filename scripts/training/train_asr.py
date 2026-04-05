#!/usr/bin/env python3
"""
ASR Fine-Tuning: Whisper large-v3 + LoRA on Armenian Data — Phase 2a

Implements efficient fine-tuning for Armenian speech recognition:
  1. Load Whisper large-v3 (frozen base) + LoRA adapter
  2. Train on Common Voice (seed) + YouTube data (scale)
  3. Evaluate on Common Voice test set (benchmark)
  4. Save best checkpoint + ONNX export

Usage:
    # Train on Common Voice only (quick validation)
    python scripts/training/train_asr.py --dataset-type common_voice --max-train-samples 5000

    # Full training on merged dataset
    python scripts/training/train_asr.py --dataset-type merged --output-dir models/asr/whisper-hy

    # Resume from checkpoint
    python scripts/training/train_asr.py --resume-from-checkpoint models/asr/whisper-hy/checkpoint_best
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.helpers import free_gpu_memory, timer
from src.training_utils import (
    AudioPreprocessor,
    DataCollatorASRWithPadding,
    MetricsComputer,
    CheckpointManager,
    TrainingProgressTracker,
    get_linear_schedule_with_warmup,
    load_jsonl_manifest,
)


# ============================================================================
# Dataset loading
# ============================================================================

class ASRDatasetLoader:
    """Load ASR datasets from manifests."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load_common_voice(self, manifest_dir: Path) -> dict:
        """Load Common Voice splits."""
        splits = {}
        for split_name in ["train", "validation", "test"]:
            manifest_file = manifest_dir / f"{split_name}.jsonl"
            if manifest_file.exists():
                splits[split_name] = load_jsonl_manifest(manifest_file)
                logger.info("Common Voice {}: {} samples", split_name, len(splits[split_name]))
        return splits

    def load_youtube(self, quality_bucket_dir: Path) -> dict:
        """Load YouTube data by quality tier."""
        splits = {}
        for tier in ["gold", "silver", "bronze"]:
            tier_file = quality_bucket_dir / f"{tier}.jsonl"
            if tier_file.exists():
                entries = load_jsonl_manifest(tier_file)
                splits[tier] = entries
                logger.info("YouTube {}: {} samples", tier, len(entries))
        return splits

    def load_merged(self, split_dir: Path) -> dict:
        """Load pre-merged dataset splits."""
        splits = {}
        for split in ["train", "val", "test"]:
            split_file = split_dir / f"{split}.jsonl"
            if split_file.exists():
                splits[split] = load_jsonl_manifest(split_file)
                logger.info("Merged {}: {} samples", split, len(splits[split]))
        return splits

    def create_hf_dataset(self, entries: list[dict]):
        """Create HuggingFace dataset from entries."""
        from datasets import Dataset

        if not entries:
            return None

        dataset = Dataset.from_list(entries)
        return dataset


# ============================================================================
# Model setup
# ============================================================================

def setup_whisper_lora(
    model_id: str = "openai/whisper-large-v3",
    lora_config: dict = None,
) -> tuple:
    """Load Whisper model with LoRA adapter."""
    if lora_config is None:
        lora_config = {}

    # Normalize keys: colab profile uses lora_r, standard uses r
    lora_r = lora_config.get("r", lora_config.get("lora_r", 32))
    lora_alpha = lora_config.get("lora_alpha", 64)
    lora_dropout = lora_config.get("lora_dropout", 0.05)
    lora_bias = lora_config.get("bias", "none")
    lora_target = lora_config.get("target_modules", ["q", "v"])

    logger.info("Loading {} model...", model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,  # 8-bit quantization for memory efficiency
    )

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    # Add LoRA adapter
    logger.info("Adding LoRA adapter: r={}, alpha={}", lora_r, lora_alpha)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=lora_target,
    )

    model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info("Trainable params: {} / {} ({:.2f}%)",
                trainable_params, total_params, 100 * trainable_params / total_params)

    return model, peft_config


# ============================================================================
# Preprocessing function
# ============================================================================

def preprocess_function(
    examples: dict,
    feature_extractor,
    tokenizer,
    sample_rate: int = 16000,
) -> dict:
    """Preprocess audio + text."""
    audio_paths = examples["audio_path"]
    texts = examples.get("text", examples.get("text_clean", []))

    # Load audio
    audio_arrays = []
    for audio_path in audio_paths:
        try:
            preprocessor = AudioPreprocessor(sample_rate)
            audio_dict = preprocessor.load_and_preprocess(audio_path)
            audio_arrays.append(audio_dict["input_values"])
        except Exception as e:
            logger.warning("Failed to load {}: {}", audio_path, e)
            audio_arrays.append(np.zeros(sample_rate * 5))  # Fallback: 5sec silence

    # Extract features
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=sample_rate,
        return_attention_mask=True,
    )

    # Tokenize text
    input_ids = tokenizer(texts, max_length=512, truncation=True).input_ids

    return {
        "input_features": inputs.input_features,
        "input_length": inputs.attention_mask.sum(axis=1),
        "labels": input_ids,
    }


# ============================================================================
# Evaluation
# ============================================================================

def compute_metrics(pred, feature_extractor, tokenizer):
    """Compute WER metric during training."""
    predictions = pred.predictions
    label_ids = pred.label_ids

    # Generate predictions
    pred_ids = np.argmax(predictions, axis=-1)
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = MetricsComputer.compute_wer(pred_str, label_str)
    cer = MetricsComputer.compute_cer(pred_str, label_str)

    return {"wer": wer, "cer": cer}


# ============================================================================
# Main training script
# ============================================================================

def train_asr(
    train_dataset,
    eval_dataset,
    model,
    feature_extractor,
    tokenizer,
    training_args: dict,
    output_dir: Path,
):
    """Train ASR model."""
    logger.info("Starting ASR training...")
    logger.info("Train samples: {}", len(train_dataset))
    logger.info("Eval samples: {}", len(eval_dataset))

    # Data collator for already-preprocessed features.
    # The dataset was already .map()-ed to contain input_features and labels,
    # so we just need to pad labels (features are fixed-size from Whisper).
    from dataclasses import dataclass

    @dataclass
    class PreprocessedASRCollator:
        pad_token_id: int = -100

        def __call__(self, batch):
            import torch as _torch

            input_features = _torch.tensor(
                [s["input_features"] for s in batch], dtype=_torch.float32
            )
            label_lists = [s["labels"] for s in batch]
            max_len = max(len(l) for l in label_lists)
            padded = [l + [self.pad_token_id] * (max_len - len(l)) for l in label_lists]
            labels = _torch.tensor(padded, dtype=_torch.long)
            return {"input_features": input_features, "labels": labels}

    data_collator = PreprocessedASRCollator(pad_token_id=-100)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=training_args.get("batch_size", 16),
            per_device_eval_batch_size=training_args.get("eval_batch_size", 32),
            gradient_accumulation_steps=training_args.get("gradient_accumulation", 4),
            learning_rate=training_args.get("learning_rate", 1e-4),
            num_train_epochs=training_args.get("epochs", 30),
            warmup_steps=training_args.get("warmup_steps", 500),
            save_strategy="steps",
            save_steps=training_args.get("save_steps", 1000),
            eval_strategy="steps",
            eval_steps=training_args.get("eval_steps", 500),
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            bf16=torch.cuda.get_device_capability()[0] >= 8,  # bfloat16 if available
            fp16=torch.cuda.get_device_capability()[0] < 8,   # float16 fallback
            logging_steps=100,
            logging_dir=str(output_dir / "logs"),
            dataloader_num_workers=training_args.get("dataloader_workers", 2),
            remove_unused_columns=False,
            label_names=["labels"],
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, feature_extractor, tokenizer),
    )

    # Train
    with timer("ASR training"):
        train_result = trainer.train()

    # Save final model
    model.save_pretrained(str(output_dir / "final_model"))
    logger.info("Saved final model to {}", output_dir / "final_model")

    # Results
    metrics = {
        "train_loss": round(train_result.training_loss, 4),
        "train_steps": train_result.global_step,
    }

    # Eval
    eval_results = trainer.evaluate()
    metrics.update({f"eval_{k}": v for k, v in eval_results.items()})

    logger.info("Training complete!")
    logger.info("Final metrics: {}", json.dumps(metrics, indent=2))

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper Armenian ASR Fine-Tuning")
    parser.add_argument(
        "--dataset-type",
        choices=["common_voice", "youtube", "merged"],
        default="common_voice",
    )
    parser.add_argument("--cv-dir", type=str, default="data/common_voice/manifests")
    parser.add_argument("--yt-dir", type=str, default="data/youtube_crawl/quality_buckets")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="models/asr/whisper-hy")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config-override", type=str, default=None)

    args = parser.parse_args()
    setup_logger()

    # Load config
    config = load_config(config_path=args.config, override_path=args.config_override)

    asr_config = config.get("asr", {})
    training_config = config.get("training", {}).get("asr", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup model
    model, peft_config = setup_whisper_lora(
        model_id=f"openai/whisper-{asr_config['whisper']['model']}",
        lora_config=training_config.copy(),
    )

    # Load processors
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        f"openai/whisper-{asr_config['whisper']['model']}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        f"openai/whisper-{asr_config['whisper']['model']}"
    )
    tokenizer.language = "Armenian"
    tokenizer.task = "transcribe"

    # Load datasets
    loader = ASRDatasetLoader(sample_rate=16000)

    if args.dataset_type == "common_voice":
        splits = loader.load_common_voice(Path(args.cv_dir))
        train_entries = splits.get("train", [])
        eval_entries = splits.get("validation", [])
    elif args.dataset_type == "youtube":
        yt_splits = loader.load_youtube(Path(args.yt_dir))
        train_entries = yt_splits.get("gold", [])
        eval_entries = yt_splits.get("silver", [])
    else:  # merged
        splits = loader.load_merged(Path(args.splits_dir))
        train_entries = splits.get("train", [])
        eval_entries = splits.get("val", [])

    # Limit samples for testing
    if args.max_train_samples:
        train_entries = train_entries[:args.max_train_samples]
    if args.max_eval_samples:
        eval_entries = eval_entries[:args.max_eval_samples]

    # Create HF datasets
    train_dataset = loader.create_hf_dataset(train_entries)
    eval_dataset = loader.create_hf_dataset(eval_entries)

    if train_dataset is None or eval_dataset is None:
        logger.error("Failed to load datasets")
        sys.exit(1)

    # Preprocess
    logger.info("Preprocessing dataset...")
    def preprocess_fn(examples):
        return preprocess_function(
            examples,
            feature_extractor,
            tokenizer,
            sample_rate=16000,
        )

    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=32,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=32,
        remove_columns=eval_dataset.column_names,
    )

    # Train
    train_asr(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        training_args=training_config,
        output_dir=output_dir,
    )

    logger.info("ASR training complete: {}", output_dir)


if __name__ == "__main__":
    main()
