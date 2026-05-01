#!/usr/bin/env python3
"""Package local model folders for release and optional Hugging Face upload."""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


MODEL_SPECS = {
    "asr": {
        "default_path": "models/asr/whisper-hy-full",
        "task": "automatic-speech-recognition",
        "library": "transformers",
        "pipeline_tag": "automatic-speech-recognition",
        "tags": ["armenian", "asr", "whisper", "lora"],
    },
    "tts": {
        "default_path": "models/tts/fish-speech-hy",
        "task": "text-to-speech",
        "library": "transformers",
        "pipeline_tag": "text-to-speech",
        "tags": ["armenian", "tts", "voice-cloning", "fish-speech"],
    },
    "translation": {
        "default_path": "models/translation/seamless-m4t-v2-large",
        "task": "translation",
        "library": "transformers",
        "pipeline_tag": "translation",
        "tags": ["armenian", "translation", "seamless-m4t"],
    },
    "lipsync": {
        "default_path": "models/lipsync/MuseTalk",
        "task": "video-to-video",
        "library": "other",
        "pipeline_tag": "video-to-video",
        "tags": ["armenian", "lipsync", "musetalk"],
    },
}


def build_model_card(model_name: str, source_dir: Path, profile: str | None, spec: dict) -> str:
    """Create a concise model card for packaged or published artifacts."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tags = "\n".join(f"- {tag}" for tag in spec["tags"])
    profile_line = f"- Export profile: `{profile}`\n" if profile else ""
    return f"""---
library_name: {spec['library']}
pipeline_tag: {spec['pipeline_tag']}
tags:
{tags}
license: apache-2.0
---

# {model_name}

Packaged from the Armenian Video Dubbing project.

## Source

- Local source directory: `{source_dir}`
{profile_line}- Generated at: `{generated_at}`

## Notes

- This repository was exported from local training artifacts.
- Review model outputs before using in production.
- For end-to-end dubbing, pair this artifact with the matching project configuration profile.
"""


def write_manifest(target_dir: Path, model_name: str, source_dir: Path, profile: str | None, repo_id: str | None):
    """Write a lightweight release manifest for reproducibility."""
    manifest = {
        "model_name": model_name,
        "source_dir": str(source_dir.resolve()),
        "profile": profile,
        "repo_id": repo_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(target_dir / "export_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def stage_model(source_dir: Path, target_dir: Path, model_name: str, profile: str | None, repo_id: str | None):
    """Copy a local model directory into a clean release bundle."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Model source directory not found: {source_dir}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)

    spec = MODEL_SPECS[model_name]
    readme_path = target_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            build_model_card(model_name, source_dir, profile, spec),
            encoding="utf-8",
        )

    write_manifest(target_dir, model_name, source_dir, profile, repo_id)
    logger.info("Staged {} model -> {}", model_name, target_dir)


def upload_to_hub(local_dir: Path, repo_id: str, token: str, private: bool, commit_message: str):
    """Create or reuse a model repo and upload the full folder."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message=commit_message,
        token=token,
    )
    logger.info("Uploaded {} -> https://huggingface.co/{}", local_dir.name, repo_id)


def resolve_model_path(args, model_name: str) -> Path | None:
    attr_name = f"{model_name}_model"
    value = getattr(args, attr_name, None)
    if value:
        return Path(value)
    if model_name in args.models:
        return Path(MODEL_SPECS[model_name]["default_path"])
    return None


def main():
    parser = argparse.ArgumentParser(description="Package model folders and optionally publish them to Hugging Face")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS.keys()),
        default=["asr", "tts", "translation"],
        help="Models to package or publish",
    )
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--tts-model", default=None)
    parser.add_argument("--translation-model", default=None)
    parser.add_argument("--lipsync-model", default=None)
    parser.add_argument("--output-dir", default="models/releases")
    parser.add_argument("--profile", default=None, help="Config profile used for the trained/exported artifacts")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload staged folders to Hugging Face model repos")
    parser.add_argument("--hf-namespace", default=None, help="Hugging Face namespace, for example your username or org")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token; defaults to HF_TOKEN env var")
    parser.add_argument("--repo-prefix", default="armenian-video-dubbing", help="Prefix for generated model repo names")
    parser.add_argument("--private", action="store_true", help="Create private model repos")
    parser.add_argument("--commit-message", default="Upload exported model bundle")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.push_to_hub and not hf_token:
        raise RuntimeError("HF token required: pass --hf-token or set HF_TOKEN")
    if args.push_to_hub and not args.hf_namespace:
        raise RuntimeError("HF namespace required when using --push-to-hub")

    staged = []
    for model_name in args.models:
        source_dir = resolve_model_path(args, model_name)
        if source_dir is None:
            continue

        repo_id = None
        if args.hf_namespace:
            repo_id = f"{args.hf_namespace}/{args.repo_prefix}-{model_name}"

        target_dir = output_root / model_name
        stage_model(source_dir, target_dir, model_name, args.profile, repo_id)
        staged.append((model_name, target_dir, repo_id))

    if args.push_to_hub:
        for model_name, target_dir, repo_id in staged:
            if repo_id is None:
                repo_id = f"{args.hf_namespace}/{args.repo_prefix}-{model_name}"
            upload_to_hub(
                local_dir=target_dir,
                repo_id=repo_id,
                token=hf_token,
                private=args.private,
                commit_message=args.commit_message,
            )

    logger.info("Completed export workflow for {} model(s)", len(staged))


if __name__ == "__main__":
    main()
