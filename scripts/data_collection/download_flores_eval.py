#!/usr/bin/env python3
"""
Download FLORES-200 English→Armenian text pairs for translation evaluation.

This helper writes JSONL files that are directly compatible with
scripts/training/evaluate_translation.py, which expects at least:
  - text
  - reference_text

Usage:
    python scripts/data_collection/download_flores_eval.py
    python scripts/data_collection/download_flores_eval.py --max-devtest 200
    python scripts/data_collection/download_flores_eval.py --output-dir data/flores_hye
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_flores_eval(
    output_dir: Path,
    pair_config: str = "eng_Latn-hye_Armn",
    max_dev: int | None = None,
    max_devtest: int | None = None,
):
    """Download FLORES paired text data and export evaluator-friendly JSONL files."""
    from datasets import load_dataset

    source_key, target_key = pair_config.split("-", 1)
    split_limits = {
        "dev": max_dev,
        "devtest": max_devtest,
    }

    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, int]] = {}
    combined_entries = []

    for split_name, limit in split_limits.items():
        logger.info("Downloading FLORES split='{}' config='{}'", split_name, pair_config)
        ds = load_dataset(
            "facebook/flores",
            pair_config,
            split=split_name,
            trust_remote_code=True,
        )
        manifest_path = manifest_dir / f"{split_name}.jsonl"

        entries = []
        for idx, example in enumerate(ds):
            if limit is not None and len(entries) >= limit:
                break

            source_text = (example.get(f"sentence_{source_key}") or "").strip()
            reference_text = (example.get(f"sentence_{target_key}") or "").strip()
            if not source_text or not reference_text:
                continue

            entry = {
                "id": int(example.get("id", idx)),
                "text": source_text,
                "reference_text": reference_text,
                "split": split_name,
                "source_lang": source_key,
                "target_lang": target_key,
                "domain": example.get("domain"),
                "topic": example.get("topic"),
                "url": example.get("URL"),
                "has_image": example.get("has_image"),
                "has_hyperlink": example.get("has_hyperlink"),
                "source": "flores",
            }
            entries.append(entry)
            combined_entries.append(entry)

        with open(manifest_path, "w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        stats[split_name] = {"count": len(entries)}
        logger.info("  {} -> {} samples -> {}", split_name, len(entries), manifest_path)

    combined_path = manifest_dir / "combined.jsonl"
    with open(combined_path, "w", encoding="utf-8") as handle:
        for entry in combined_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats["combined"] = {"count": len(combined_entries)}

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    logger.info("Done. Stats written to {}", stats_path)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download FLORES English-Armenian evaluation data")
    parser.add_argument("--output-dir", default="data/flores_hye", help="Output directory")
    parser.add_argument(
        "--pair-config",
        default="eng_Latn-hye_Armn",
        help="FLORES paired config, default is English to Eastern Armenian",
    )
    parser.add_argument("--max-dev", type=int, default=None)
    parser.add_argument("--max-devtest", type=int, default=None)
    args = parser.parse_args()

    from src.utils.logger import setup_logger

    setup_logger()

    download_flores_eval(
        output_dir=Path(args.output_dir),
        pair_config=args.pair_config,
        max_dev=args.max_dev,
        max_devtest=args.max_devtest,
    )


if __name__ == "__main__":
    main()