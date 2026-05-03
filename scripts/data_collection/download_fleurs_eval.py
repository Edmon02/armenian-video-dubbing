#!/usr/bin/env python3
"""
Download Armenian FLEURS splits for clean ASR evaluation and small-footprint trials.

FLEURS is a useful complement to Common Voice in this repository:
  - Common Voice is the best first seed dataset for Armenian ASR fine-tuning.
  - FLEURS provides a smaller, cleaner held-out benchmark for repeatable testing.

Usage:
    python scripts/data_collection/download_fleurs_eval.py
    python scripts/data_collection/download_fleurs_eval.py --max-test 100
    python scripts/data_collection/download_fleurs_eval.py --lang-config hy_am --output-dir data/fleurs_hy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_fleurs_eval(
    output_dir: Path,
    lang_config: str = "hy_am",
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
):
    """Download FLEURS Armenian data and export manifests compatible with repo scripts."""
    from datasets import load_dataset

    split_limits = {
        "train": max_train,
        "validation": max_val,
        "test": max_test,
    }

    audio_dir = output_dir / "audio"
    manifest_dir = output_dir / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, float | int]] = {}

    for split_name, limit in split_limits.items():
        logger.info("Downloading FLEURS split='{}' config='{}'", split_name, lang_config)
        ds = load_dataset(
            "google/fleurs",
            lang_config,
            split=split_name,
            streaming=True,
        )

        split_audio_dir = audio_dir / split_name
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{split_name}.jsonl"

        entries = []
        for idx, example in enumerate(ds):
            if limit is not None and len(entries) >= limit:
                break

            transcription = (example.get("transcription") or "").strip()
            raw_transcription = (example.get("raw_transcription") or transcription).strip()
            audio = example.get("audio")
            if not transcription or audio is None:
                continue

            try:
                import soundfile as sf

                audio_array = np.asarray(audio["array"], dtype=np.float32)
                sample_rate = int(audio["sampling_rate"])

                if sample_rate != 16000:
                    import librosa

                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sample_rate,
                        target_sr=16000,
                    )
                    sample_rate = 16000

                duration = len(audio_array) / sample_rate
                if duration < 0.5 or duration > 40:
                    continue

                clip_id = example.get("id", idx)
                audio_path = split_audio_dir / f"fleurs_{lang_config}_{split_name}_{clip_id}.wav"
                sf.write(str(audio_path), audio_array, sample_rate)

                gender = example.get("gender")
                if isinstance(gender, int):
                    if gender == 0:
                        gender = "female"
                    elif gender == 1:
                        gender = "male"

                entries.append(
                    {
                        "audio_path": str(audio_path.resolve()),
                        "text": transcription,
                        "text_raw": raw_transcription,
                        "duration_sec": round(duration, 3),
                        "sample_rate": sample_rate,
                        "split": split_name,
                        "source": "fleurs",
                        "language": lang_config,
                        "gender": gender,
                    }
                )
            except Exception as exc:
                logger.debug("Skipping FLEURS sample {}: {}", idx, exc)
                continue

        with open(manifest_path, "w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        total_sec = sum(float(entry["duration_sec"]) for entry in entries)
        stats[split_name] = {
            "count": len(entries),
            "seconds": round(total_sec, 1),
        }
        logger.info(
            "  {} -> {} samples, {:.1f}s total -> {}",
            split_name,
            len(entries),
            total_sec,
            manifest_path,
        )

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    logger.info("Done. Stats written to {}", stats_path)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download Armenian FLEURS evaluation data")
    parser.add_argument("--output-dir", default="data/fleurs_hy", help="Output directory")
    parser.add_argument("--lang-config", default="hy_am", help="FLEURS language config")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    from src.utils.logger import setup_logger

    setup_logger()

    download_fleurs_eval(
        output_dir=Path(args.output_dir),
        lang_config=args.lang_config,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
    )


if __name__ == "__main__":
    main()