#!/usr/bin/env python3
"""
Download a tiny subset of Common Voice hy-AM for Colab smoke testing.

Downloads only a handful of samples (default 80 train + 20 validation)
to keep Colab storage and time minimal.  Produces JSONL manifests
compatible with scripts/training/train_asr.py.

Usage:
    python scripts/data_collection/download_cv_tiny.py
    python scripts/data_collection/download_cv_tiny.py --max-train 64 --max-val 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger


def download_tiny_cv(
    output_dir: Path,
    max_train: int = 80,
    max_val: int = 20,
    version: str = "17_0",
):
    """Download a tiny slice of Common Voice hy-AM and write manifests."""
    from datasets import load_dataset

    dataset_name = f"mozilla-foundation/common_voice_{version}"
    logger.info("Loading {} hy-AM (streaming)...", dataset_name)

    audio_dir = output_dir / "audio"
    manifest_dir = output_dir / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    splits_config = {
        "train": max_train,
        "validation": max_val,
    }

    stats = {}

    for split_name, max_samples in splits_config.items():
        logger.info("Downloading split='{}' (up to {} samples)...", split_name, max_samples)

        try:
            ds = load_dataset(
                dataset_name,
                "hy-AM",
                split=split_name,
                trust_remote_code=True,
                streaming=True,
            )
        except Exception as e:
            logger.error("Failed to load split '{}': {}", split_name, e)
            continue

        split_audio_dir = audio_dir / split_name
        split_audio_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for idx, example in enumerate(ds):
            if idx >= max_samples:
                break

            sentence = (example.get("sentence") or "").strip()
            audio = example.get("audio")
            if not sentence or audio is None:
                continue

            try:
                import soundfile as sf

                audio_array = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]

                # Resample to 16 kHz if needed
                if sr != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    sr = 16000

                duration = len(audio_array) / sr
                if duration < 0.5 or duration > 30:
                    continue

                clip_id = example.get("path", f"cv_{idx:07d}")
                if isinstance(clip_id, str):
                    clip_id = Path(clip_id).stem
                else:
                    clip_id = f"cv_{idx:07d}"

                audio_path = split_audio_dir / f"{clip_id}.wav"
                sf.write(str(audio_path), audio_array, sr)

                entries.append({
                    "audio_path": str(audio_path.resolve()),
                    "text": sentence,
                    "duration_sec": round(duration, 3),
                    "sample_rate": sr,
                    "split": split_name,
                    "source": "common_voice",
                })
            except Exception as e:
                logger.debug("Skipping sample {}: {}", idx, e)
                continue

        manifest_path = manifest_dir / f"{split_name}.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        total_sec = sum(e["duration_sec"] for e in entries)
        stats[split_name] = {"count": len(entries), "seconds": round(total_sec, 1)}
        logger.info(
            "  {} -> {} samples, {:.1f}s total -> {}",
            split_name, len(entries), total_sec, manifest_path,
        )

    logger.info("Done. Stats: {}", json.dumps(stats))
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download tiny Common Voice hy-AM subset")
    parser.add_argument("--output-dir", default="data/common_voice", help="Output directory")
    parser.add_argument("--max-train", type=int, default=80)
    parser.add_argument("--max-val", type=int, default=20)
    parser.add_argument("--version", default="17_0", help="Common Voice dataset version tag")
    args = parser.parse_args()

    from src.utils.logger import setup_logger
    setup_logger()

    download_tiny_cv(
        output_dir=Path(args.output_dir),
        max_train=args.max_train,
        max_val=args.max_val,
        version=args.version,
    )


if __name__ == "__main__":
    main()
