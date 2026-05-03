#!/usr/bin/env python3
"""
Label Studio Setup for Armenian ASR Validation — Phase 1c

Implements the validation loop from "Scaling Armenian ASR" paper:
  1. Deploy Label Studio with ASR annotation interface
  2. Import silver/bronze-tier segments for human validation
  3. Export validated transcriptions back to training pipeline
  4. Compute inter-annotator agreement

Usage:
    python scripts/data_collection/labelstudio_setup.py --action setup
    python scripts/data_collection/labelstudio_setup.py --action import --tier silver
    python scripts/data_collection/labelstudio_setup.py --action export
"""

import argparse
import json
import os
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


# Label Studio annotation interface for ASR validation
LABELING_CONFIG_XML = """
<View>
  <Header value="Armenian ASR Transcription Validation" />
  <Text name="info" value="Listen to the audio and correct the transcription if needed." />

  <View style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
    <Text name="meta" value="Segment: $segment_id | Duration: $duration_sec s | Quality: $quality_tier" />
  </View>

  <Audio name="audio" value="$audio_url" />

  <Header value="Auto-generated Transcription (edit below):" size="4" />
  <TextArea name="transcription"
            toName="audio"
            value="$transcription_text"
            rows="3"
            editable="true"
            maxSubmissions="1"
            showSubmitButton="false" />

  <Header value="Quality Assessment:" size="4" />
  <Choices name="quality" toName="audio" choice="single-radio" showInline="true">
    <Choice value="correct" alias="correct" />
    <Choice value="minor_errors" alias="minor" />
    <Choice value="major_errors" alias="major" />
    <Choice value="unusable" alias="unusable" />
  </Choices>

  <Header value="Audio Quality:" size="4" />
  <Choices name="audio_quality" toName="audio" choice="single-radio" showInline="true">
    <Choice value="clean" alias="clean" />
    <Choice value="moderate_noise" alias="moderate" />
    <Choice value="noisy" alias="noisy" />
    <Choice value="music_overlap" alias="music" />
  </Choices>

  <Header value="Dialect (if identifiable):" size="4" />
  <Choices name="dialect" toName="audio" choice="single-radio" showInline="true" required="false">
    <Choice value="eastern_armenian" alias="east" />
    <Choice value="western_armenian" alias="west" />
    <Choice value="uncertain" alias="uncertain" />
  </Choices>
</View>
"""


class LabelStudioManager:
    """Manage Label Studio project for ASR validation."""

    def __init__(
        self,
        ls_url: str = "http://localhost:8080",
        api_key: str | None = None,
    ):
        self.ls_url = ls_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LABEL_STUDIO_API_KEY", "")

    def _sdk_client(self):
        """Get Label Studio SDK client."""
        try:
            from label_studio_sdk import Client
            return Client(url=self.ls_url, api_key=self.api_key)
        except ImportError:
            logger.error("label-studio-sdk not installed. Run: pip install label-studio-sdk")
            sys.exit(1)
        except Exception as e:
            logger.error("Cannot connect to Label Studio at {}: {}", self.ls_url, e)
            logger.info("Start Label Studio: docker compose up label-studio")
            sys.exit(1)

    def setup_project(self, project_name: str = "Armenian ASR Validation") -> int:
        """Create Label Studio project with ASR annotation config.

        Returns:
            Project ID.
        """
        client = self._sdk_client()

        # Check if project already exists
        projects = client.get_projects()
        for p in projects:
            if p.title == project_name:
                logger.info("Project '{}' already exists (ID: {})", project_name, p.id)
                return p.id

        # Create new project
        project = client.create_project(
            title=project_name,
            description="Validate and correct Armenian ASR transcriptions from YouTube crawl data.",
            label_config=LABELING_CONFIG_XML,
        )

        logger.info("Created project '{}' (ID: {})", project_name, project.id)
        return project.id

    def import_tasks(
        self,
        project_id: int,
        manifest_path: Path,
        audio_serve_prefix: str = "/data/local-files/?d=",
        max_tasks: int | None = None,
    ) -> int:
        """Import segments from manifest as Label Studio tasks.

        Args:
            project_id: Label Studio project ID.
            manifest_path: Path to JSONL manifest (quality bucketed).
            audio_serve_prefix: URL prefix for serving audio files.
            max_tasks: Maximum tasks to import (None = all).

        Returns:
            Number of tasks imported.
        """
        client = self._sdk_client()
        project = client.get_project(project_id)

        tasks = []
        with open(manifest_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                audio_path = entry.get("audio_path", "")
                text = entry.get("transcription", {}).get("text_clean", "")
                segment_id = entry.get("segment_id", "")
                duration = entry.get("duration_sec", 0)
                tier = entry.get("quality_tier", "unknown")

                # Build audio URL for Label Studio
                audio_url = f"{audio_serve_prefix}{audio_path}"

                task = {
                    "data": {
                        "audio_url": audio_url,
                        "transcription_text": text,
                        "segment_id": segment_id,
                        "duration_sec": str(round(duration, 1)),
                        "quality_tier": tier,
                    },
                    "meta": {
                        "video_id": entry.get("video_id", ""),
                        "snr_db": entry.get("snr_db", 0),
                        "avg_logprob": entry.get("transcription", {}).get("avg_logprob", 0),
                    },
                }
                tasks.append(task)

                if max_tasks and len(tasks) >= max_tasks:
                    break

        if not tasks:
            logger.warning("No tasks to import from {}", manifest_path)
            return 0

        # Import in batches of 100
        batch_size = 100
        total = 0
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            project.import_tasks(batch)
            total += len(batch)
            logger.info("Imported {}/{} tasks", total, len(tasks))

        logger.info("Total tasks imported: {}", total)
        return total

    def export_annotations(
        self,
        project_id: int,
        output_path: Path,
    ) -> int:
        """Export completed annotations to JSONL manifest.

        Returns:
            Number of annotated entries exported.
        """
        client = self._sdk_client()
        project = client.get_project(project_id)

        # Get completed tasks
        tasks = project.get_labeled_tasks()

        logger.info("Exporting {} annotated tasks...", len(tasks))

        count = 0
        with open(output_path, "w") as f:
            for task in tasks:
                data = task.get("data", {})
                annotations = task.get("annotations", [])

                if not annotations:
                    continue

                # Take last annotation (most recent)
                ann = annotations[-1]
                results = {r.get("from_name"): r.get("value", {}) for r in ann.get("result", [])}

                entry = {
                    "segment_id": data.get("segment_id", ""),
                    "audio_path": data.get("audio_url", "").replace("/data/local-files/?d=", ""),
                    "original_text": data.get("transcription_text", ""),
                    "validated_text": results.get("transcription", {}).get("text", [""])[0] if "transcription" in results else data.get("transcription_text", ""),
                    "quality_label": results.get("quality", {}).get("choices", [""])[0] if "quality" in results else "",
                    "audio_quality": results.get("audio_quality", {}).get("choices", [""])[0] if "audio_quality" in results else "",
                    "dialect": results.get("dialect", {}).get("choices", [""])[0] if "dialect" in results else "",
                    "annotator": ann.get("completed_by", {}).get("email", "unknown"),
                    "annotation_id": ann.get("id"),
                }

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        logger.info("Exported {} annotations to {}", count, output_path)
        return count

    def compute_agreement(self, output_path: Path) -> dict:
        """Compute inter-annotator agreement statistics."""
        from collections import Counter

        entries = []
        with open(output_path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Quality distribution
        quality_dist = Counter(e.get("quality_label", "") for e in entries)
        audio_dist = Counter(e.get("audio_quality", "") for e in entries)
        dialect_dist = Counter(e.get("dialect", "") for e in entries)

        # Text edit rate (how much annotators changed transcriptions)
        edit_rates = []
        for e in entries:
            orig = e.get("original_text", "")
            validated = e.get("validated_text", "")
            if orig and validated:
                # Simple character edit distance ratio
                max_len = max(len(orig), len(validated), 1)
                changes = sum(1 for a, b in zip(orig, validated) if a != b) + abs(len(orig) - len(validated))
                edit_rates.append(changes / max_len)

        stats = {
            "total_annotations": len(entries),
            "quality_distribution": dict(quality_dist),
            "audio_quality_distribution": dict(audio_dist),
            "dialect_distribution": dict(dialect_dist),
            "avg_edit_rate": round(sum(edit_rates) / max(len(edit_rates), 1), 4),
            "correct_fraction": round(quality_dist.get("correct", 0) / max(len(entries), 1), 4),
        }

        logger.info("Annotation stats:")
        for k, v in stats.items():
            logger.info("  {}: {}", k, v)

        return stats


# ============================================================================
# Annotation Guide Generator
# ============================================================================

def generate_annotation_guide(output_path: Path):
    """Generate a PDF-ready annotation guide for Armenian speakers."""
    guide = """
# Armenian ASR Validation — Annotation Guide

## Task Overview
You will listen to short audio clips (1-30 seconds) of Armenian speech and:
1. Verify/correct the auto-generated transcription
2. Rate the quality of the transcription
3. Rate the audio quality
4. Identify the dialect (Eastern/Western Armenian)

## Transcription Rules

### Spelling
- Use standard Eastern Armenian orthography (Reformed/Soviet)
- Keep ե for /je/ at word-initial position
- Use standard punctuation: ։ (verjaket), ՝ (boot), ՜ (patgaman), ՞ (harcakan)

### Numbers & Abbreviations
- Write numbers as words: «երկու հազար» not «2000»
- Spell out common abbreviations: «Հայաստանի Հանրապետություն» not «ՀՀ»
- Exception: keep proper nouns/brands as-is

### Foreign Words
- Transliterate foreign words commonly used in Armenian speech
- Keep English brand names in Armenian script if commonly written that way

### Fillers & Disfluencies
- Include natural speech fillers: «էէէ», «այ», «ըըը»
- Mark unintelligible portions with [unintelligible]
- Mark overlap with [overlap]

## Quality Ratings

| Rating | Description |
|--------|-------------|
| correct | Transcription is >95% accurate, only minor spelling fixes needed |
| minor_errors | 1-3 word errors, meaning preserved |
| major_errors | Multiple errors, meaning partially lost |
| unusable | Transcription is mostly wrong or audio is not Armenian speech |

## Audio Quality Ratings

| Rating | Description |
|--------|-------------|
| clean | Clear speech, minimal background noise |
| moderate_noise | Some background noise but speech is clearly audible |
| noisy | Significant noise, some words hard to hear |
| music_overlap | Background music overlaps with speech |

## Dialect Guide

| Dialect | Key markers |
|---------|-------------|
| Eastern Armenian | Standard Yerevan pronunciation, -ում verb endings |
| Western Armenian | Istanbul/diaspora pronunciation, -կ verb endings, different consonant voicing |
| Uncertain | Cannot reliably determine |

## Quality Targets
- Aim for at least 50 annotations per hour
- Flag any non-Armenian content immediately (mark as unusable)
- When in doubt about a word, listen 2-3 times before marking
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(guide)

    logger.info("Annotation guide written to {}", output_path)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Label Studio ASR Validation Manager")
    parser.add_argument(
        "--action",
        choices=["setup", "import", "export", "guide", "stats"],
        required=True,
    )
    parser.add_argument("--ls-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument("--tier", default="silver", help="Quality tier to import (silver, bronze, gold)")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output-dir", default="data/youtube_crawl")

    args = parser.parse_args()
    setup_logger()

    output_dir = Path(args.output_dir)
    manager = LabelStudioManager(args.ls_url, args.api_key)

    if args.action == "setup":
        project_id = manager.setup_project()
        logger.info("Project ready. Use --project-id {} for import/export", project_id)

    elif args.action == "import":
        if not args.project_id:
            logger.error("--project-id required for import")
            sys.exit(1)

        manifest = output_dir / "quality_buckets" / f"{args.tier}.jsonl"
        if not manifest.exists():
            logger.error("Manifest not found: {}", manifest)
            sys.exit(1)

        manager.import_tasks(args.project_id, manifest, max_tasks=args.max_tasks)

    elif args.action == "export":
        if not args.project_id:
            logger.error("--project-id required for export")
            sys.exit(1)

        export_path = output_dir / "validated_annotations.jsonl"
        manager.export_annotations(args.project_id, export_path)
        manager.compute_agreement(export_path)

    elif args.action == "guide":
        generate_annotation_guide(output_dir / "annotation_guide.md")

    elif args.action == "stats":
        annotations_path = output_dir / "validated_annotations.jsonl"
        if not annotations_path.exists():
            logger.error("No annotations file. Export first.")
            sys.exit(1)
        manager.compute_agreement(annotations_path)


if __name__ == "__main__":
    main()
