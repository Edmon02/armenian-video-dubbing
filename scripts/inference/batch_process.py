#!/usr/bin/env python3
"""
Batch Processing for Dubbing — Phase 3e

Process multiple videos in sequence or parallel:
  - Load video list from CSV/JSON
  - Process each with specified settings
  - Collect metrics and results
  - Generate report

Usage:
    python scripts/inference/batch_process.py --input videos.csv --output results.json
    python scripts/inference/batch_process.py --input videos/ --pattern "*.mp4" --emotion happy
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.pipeline import DubbingPipeline
from src.utils.logger import setup_logger


class BatchProcessor:
    """Process multiple videos."""

    def __init__(self, dry_run: bool = False):
        self.pipeline = DubbingPipeline()
        self.dry_run = dry_run
        self.results = []

    def load_manifest(self, manifest_path: Path) -> List[Dict]:
        """Load video manifest from CSV or JSON."""
        manifest_path = Path(manifest_path)

        if manifest_path.suffix == ".csv":
            df = pd.read_csv(manifest_path)
            return df.to_dict("records")

        elif manifest_path.suffix == ".json":
            with open(manifest_path) as f:
                return json.load(f)

        elif manifest_path.is_dir():
            # Directory of videos
            videos = sorted(manifest_path.glob("*.mp4")) + sorted(manifest_path.glob("*.mkv"))
            return [{"video": str(v)} for v in videos]

        else:
            raise ValueError(f"Unsupported format: {manifest_path}")

    def process_batch(
        self,
        manifest_path: Path,
        output_dir: Path = Path("outputs/batch_dubbed"),
        emotion: str = "neutral",
        skip_lipsync: bool = False,
    ):
        """Process batch of videos."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        jobs = self.load_manifest(manifest_path)
        logger.info("Loaded {} videos", len(jobs))

        # Process each
        for i, job in enumerate(tqdm(jobs, desc="Processing batch"), 1):
            video_path = job.get("video") or job.get("path") or job.get("file")
            if not video_path:
                logger.warning("Job {}: no video path", i)
                continue

            video_path = Path(video_path)
            if not video_path.exists():
                logger.warning("Job {}: file not found - {}", i, video_path)
                continue

            # Prepare output
            output_path = output_dir / f"{video_path.stem}_dubbed_{emotion}.mp4"

            # Get options from manifest
            job_emotion = job.get("emotion", emotion)
            job_skip_lipsync = job.get("skip_lipsync", skip_lipsync)
            reference_speaker = job.get("reference_speaker")

            logger.info("[{}/{}] Processing: {}", i, len(jobs), video_path.name)

            if self.dry_run:
                logger.info("  [DRY RUN] Would process with emotion={}", job_emotion)
                self.results.append({
                    "video": str(video_path),
                    "status": "dry_run",
                    "output": str(output_path),
                })
                continue

            # Run dubbing
            try:
                result = self.pipeline.dub_video(
                    video_path=str(video_path),
                    reference_speaker_audio=reference_speaker,
                    emotion=job_emotion,
                    output_path=str(output_path),
                    skip_lipsync=job_skip_lipsync,
                    voice_consent=bool(job.get("voice_consent", False)),
                )

                self.results.append({
                    "video": str(video_path),
                    "status": result.get("status", "unknown"),
                    "output": str(output_path),
                    "emotion": job_emotion,
                    "duration_sec": result.get("duration_sec"),
                    "error": result.get("error"),
                })

                if result.get("status") == "success":
                    logger.info("  ✓ Success")
                else:
                    logger.warning("  ✗ Failed: {}", result.get("error"))

            except Exception as e:
                logger.error("  ✗ Exception: {}", e)
                self.results.append({
                    "video": str(video_path),
                    "status": "error",
                    "error": str(e),
                })

        return self.results

    def save_results(self, output_path: Path):
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info("Results saved to: {}", output_path)

        # Summary
        success = sum(1 for r in self.results if r.get("status") == "success")
        failed = sum(1 for r in self.results if r.get("status") == "error")
        total = len(self.results)

        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"  Total:    {total}")
        print(f"  Success:  {success}")
        print(f"  Failed:   {failed}")
        print(f"  Success rate: {100 * success / max(total, 1):.1f}%")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch Video Dubbing")
    parser.add_argument("--input", type=str, required=True,
                        help="Manifest (CSV/JSON) or directory of videos")
    parser.add_argument("--output-dir", type=str, default="outputs/batch_dubbed",
                        help="Output directory")
    parser.add_argument("--output-results", type=str, default="batch_results.json",
                        help="Results JSON file")
    parser.add_argument("--emotion", type=str, default="neutral",
                        choices=["neutral", "happy", "sad", "angry", "excited", "calm"])
    parser.add_argument("--skip-lipsync", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without running")

    args = parser.parse_args()
    setup_logger()

    processor = BatchProcessor(dry_run=args.dry_run)

    results = processor.process_batch(
        manifest_path=Path(args.input),
        output_dir=Path(args.output_dir),
        emotion=args.emotion,
        skip_lipsync=args.skip_lipsync,
    )

    processor.save_results(Path(args.output_results))


if __name__ == "__main__":
    main()
