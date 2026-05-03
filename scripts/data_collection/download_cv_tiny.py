#!/usr/bin/env python3
"""
Download a tiny Armenian ASR dataset for smoke testing.

The original implementation streamed Common Voice from Hugging Face. That path
no longer works because Common Voice moved to Mozilla Data Collective in late
2025. This helper now does the following:

1. Tries the legacy Hugging Face Common Voice route for backward compatibility.
2. Tries Mozilla Data Collective if you provide an MDC dataset id and API key.
3. Falls back to FLEURS Armenian so the smoke-test training path still works.

All routes write JSONL manifests compatible with scripts/training/train_asr.py.

Usage:
    python scripts/data_collection/download_cv_tiny.py
    python scripts/data_collection/download_cv_tiny.py --max-train 64 --max-val 16
    python scripts/data_collection/download_cv_tiny.py --mdc-dataset-id YOUR_MDC_DATASET_ID
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collection.download_fleurs_eval import download_fleurs_eval


def _normalize_split_name(raw_split: str | None) -> str | None:
    """Map dataset-specific split labels to the repo's expected split names."""
    if raw_split is None:
        return None

    split_name = str(raw_split).strip().lower()
    if split_name in {"train", "training"}:
        return "train"
    if split_name in {"validation", "valid", "val", "dev"}:
        return "validation"
    if split_name in {"test", "eval", "evaluation"}:
        return "test"
    return None


def _write_manifest_entries(
    output_dir: Path,
    split_name: str,
    entries: list[dict],
) -> dict[str, float | int]:
    """Write a manifest file and return simple aggregate stats."""
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"{split_name}.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_seconds = sum(float(entry["duration_sec"]) for entry in entries)
    logger.info(
        "  {} -> {} samples, {:.1f}s total -> {}",
        split_name,
        len(entries),
        total_seconds,
        manifest_path,
    )
    return {"count": len(entries), "seconds": round(total_seconds, 1)}


def _export_mdc_dataframe(
    dataset_frame,
    output_dir: Path,
    max_train: int,
    max_val: int,
) -> dict[str, dict[str, float | int]]:
    """Convert an MDC ASR dataframe into repo-compatible manifests."""
    if "audio_path" not in dataset_frame.columns or "transcription" not in dataset_frame.columns:
        raise ValueError(
            "Unsupported MDC dataset format. Expected columns 'audio_path' and 'transcription'."
        )

    import librosa
    import soundfile as sf

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    split_limits = {
        "train": max_train,
        "validation": max_val,
    }
    split_entries: dict[str, list[dict]] = {"train": [], "validation": []}

    split_column = None
    if "split" in dataset_frame.columns:
        split_column = "split"
    elif "splits" in dataset_frame.columns:
        split_column = "splits"

    candidate_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    fallback_rows: list[dict] = []

    for row in dataset_frame.to_dict(orient="records"):
        normalized_split = _normalize_split_name(row.get(split_column)) if split_column else None
        if normalized_split is None:
            fallback_rows.append(row)
        else:
            candidate_rows[normalized_split].append(row)

    if not candidate_rows["validation"] and candidate_rows["test"]:
        candidate_rows["validation"] = candidate_rows["test"]

    for split_name, limit in split_limits.items():
        rows = candidate_rows[split_name]
        if len(rows) < limit:
            deficit = limit - len(rows)
            rows = rows + fallback_rows[:deficit]
            fallback_rows = fallback_rows[deficit:]

        split_audio_dir = audio_dir / split_name
        split_audio_dir.mkdir(parents=True, exist_ok=True)

        for row_index, row in enumerate(rows):
            if len(split_entries[split_name]) >= limit:
                break

            transcription = str(row.get("transcription", "")).strip()
            audio_source = Path(str(row.get("audio_path", ""))).expanduser()
            if not transcription or not audio_source.exists():
                continue

            try:
                audio_array, sample_rate = librosa.load(str(audio_source), sr=16000, mono=True)
            except Exception as exc:
                logger.debug("Skipping MDC sample {} ({}): {}", row_index, audio_source, exc)
                continue

            duration = len(audio_array) / 16000
            if duration < 0.5 or duration > 40:
                continue

            clip_id = audio_source.stem or f"mdc_{split_name}_{row_index:07d}"
            audio_path = split_audio_dir / f"{clip_id}.wav"
            sf.write(str(audio_path), audio_array, 16000)

            split_entries[split_name].append(
                {
                    "audio_path": str(audio_path.resolve()),
                    "text": transcription,
                    "duration_sec": round(duration, 3),
                    "sample_rate": 16000,
                    "split": split_name,
                    "source": "common_voice_mdc",
                    "speaker_id": row.get("speaker_id") or row.get("client_id"),
                }
            )

    if not split_entries["train"] or not split_entries["validation"]:
        raise RuntimeError(
            "MDC dataset did not produce enough train/validation samples for the smoke test."
        )

    stats = {
        split_name: _write_manifest_entries(output_dir, split_name, entries)
        for split_name, entries in split_entries.items()
    }
    return stats


def _try_download_from_mdc(
    output_dir: Path,
    max_train: int,
    max_val: int,
    mdc_dataset_id: str,
):
    """Try loading Common Voice from Mozilla Data Collective."""
    if not os.environ.get("MDC_API_KEY"):
        raise RuntimeError(
            "MDC_API_KEY is not set. Create a Mozilla Data Collective API key first."
        )

    try:
        from datacollective import get_dataset_details, load_dataset as load_mdc_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datacollective package is not installed. Install it with: pip install datacollective"
        ) from exc

    dataset_details = get_dataset_details(mdc_dataset_id)
    logger.info(
        "Using MDC dataset: {} ({})",
        dataset_details.get("name", mdc_dataset_id),
        dataset_details.get("id", mdc_dataset_id),
    )
    dataset_frame = load_mdc_dataset(
        mdc_dataset_id,
        download_directory=str(output_dir / "mdc_cache"),
        show_progress=True,
    )
    return _export_mdc_dataframe(
        dataset_frame=dataset_frame,
        output_dir=output_dir,
        max_train=max_train,
        max_val=max_val,
    )


def download_tiny_cv(
    output_dir: Path,
    max_train: int = 80,
    max_val: int = 20,
    version: str = "17.0",
    mdc_dataset_id: str | None = None,
    fallback_source: str = "fleurs",
    fleurs_lang_config: str = "hy_am",
):
    """Download a tiny Armenian ASR slice and write training manifests."""
    from datasets import load_dataset

    # Try multiple dataset name formats — HuggingFace naming varies by version
    candidate_names = [
        f"mozilla-foundation/common_voice_{version.replace('.', '_')}",
        f"mozilla-foundation/common_voice_{version}",
    ]
    # Also try older versions as fallback
    fallback_versions = ["16.1", "16.0", "13.0"]
    for fv in fallback_versions:
        candidate_names.append(f"mozilla-foundation/common_voice_{fv.replace('.', '_')}")
        candidate_names.append(f"mozilla-foundation/common_voice_{fv}")

    dataset_name = None
    hf_failure_reasons: list[str] = []
    for name in candidate_names:
        logger.info("Trying dataset: {} ...", name)
        try:
            test_ds = load_dataset(
                name, "hy-AM", split="train", trust_remote_code=True, streaming=True,
            )
            # Verify we can actually read a sample
            next(iter(test_ds))
            dataset_name = name
            logger.info("Using dataset: {}", dataset_name)
            break
        except Exception as e:
            logger.debug("  {} did not work: {}", name, e)
            hf_failure_reasons.append(f"{name}: {e}")
            continue

    if dataset_name is not None:
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

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
            except Exception as exc:
                logger.error("Failed to load split '{}': {}", split_name, exc)
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
                    sample_rate = audio["sampling_rate"]

                    if sample_rate != 16000:
                        import librosa

                        audio_array = librosa.resample(
                            audio_array,
                            orig_sr=sample_rate,
                            target_sr=16000,
                        )
                        sample_rate = 16000

                    duration = len(audio_array) / sample_rate
                    if duration < 0.5 or duration > 30:
                        continue

                    clip_id = example.get("path", f"cv_{idx:07d}")
                    if isinstance(clip_id, str):
                        clip_id = Path(clip_id).stem
                    else:
                        clip_id = f"cv_{idx:07d}"

                    audio_path = split_audio_dir / f"{clip_id}.wav"
                    sf.write(str(audio_path), audio_array, sample_rate)

                    entries.append(
                        {
                            "audio_path": str(audio_path.resolve()),
                            "text": sentence,
                            "duration_sec": round(duration, 3),
                            "sample_rate": sample_rate,
                            "split": split_name,
                            "source": "common_voice_hf_legacy",
                        }
                    )
                except Exception as exc:
                    logger.debug("Skipping sample {}: {}", idx, exc)
                    continue

            stats[split_name] = _write_manifest_entries(output_dir, split_name, entries)

        logger.info("Done. Stats: {}", json.dumps(stats))
        return stats

    logger.warning(
        "Common Voice hy-AM is no longer published on Hugging Face. Mozilla moved it to Mozilla Data Collective."
    )

    if mdc_dataset_id:
        try:
            stats = _try_download_from_mdc(
                output_dir=output_dir,
                max_train=max_train,
                max_val=max_val,
                mdc_dataset_id=mdc_dataset_id,
            )
            logger.info("Done via Mozilla Data Collective. Stats: {}", json.dumps(stats))
            return stats
        except Exception as exc:
            logger.warning("MDC download path failed: {}", exc)

    if fallback_source == "fleurs":
        logger.warning(
            "Falling back to FLEURS Armenian so the ASR smoke test can still run. "
            "Provide --mdc-dataset-id with MDC_API_KEY to use official Common Voice instead."
        )
        stats = download_fleurs_eval(
            output_dir=output_dir,
            lang_config=fleurs_lang_config,
            max_train=max_train,
            max_val=max_val,
            max_test=max_val,
        )
        logger.info("Done via FLEURS fallback. Stats: {}", json.dumps(stats))
        return stats

    raise RuntimeError(
        "Could not load Common Voice hy-AM from legacy Hugging Face ids and fallback is disabled. "
        "Set MDC_API_KEY, accept the dataset terms on Mozilla Data Collective, and pass --mdc-dataset-id. "
        "Legacy attempts: "
        + " | ".join(hf_failure_reasons)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download a tiny Armenian ASR dataset for smoke tests"
    )
    parser.add_argument("--output-dir", default="data/common_voice", help="Output directory")
    parser.add_argument("--max-train", type=int, default=80)
    parser.add_argument("--max-val", type=int, default=20)
    parser.add_argument("--version", default="17.0", help="Common Voice dataset version tag")
    parser.add_argument(
        "--mdc-dataset-id",
        default=None,
        help="Mozilla Data Collective dataset id or slug for Armenian Common Voice",
    )
    parser.add_argument(
        "--fallback-source",
        choices=["fleurs", "none"],
        default="fleurs",
        help="Fallback dataset if Common Voice is unavailable from Hugging Face or MDC",
    )
    parser.add_argument(
        "--fleurs-lang-config",
        default="hy_am",
        help="FLEURS language config used for fallback smoke-test data",
    )
    args = parser.parse_args()

    from src.utils.logger import setup_logger
    setup_logger()

    download_tiny_cv(
        output_dir=Path(args.output_dir),
        max_train=args.max_train,
        max_val=args.max_val,
        version=args.version,
        mdc_dataset_id=args.mdc_dataset_id,
        fallback_source=args.fallback_source,
        fleurs_lang_config=args.fleurs_lang_config,
    )


if __name__ == "__main__":
    main()
