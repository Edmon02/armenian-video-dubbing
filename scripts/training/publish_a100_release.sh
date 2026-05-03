#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_NAMESPACE:-}" ]]; then
    echo "HF_NAMESPACE is required, for example: export HF_NAMESPACE=Edmon02" >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ASR_MODEL="${ASR_MODEL:-models/asr/whisper-hy-a100}"
TTS_MODEL="${TTS_MODEL:-models/tts/fish-speech-hy-lightning-smoke}"
TRANSLATION_MODEL="${TRANSLATION_MODEL:-models/translation/seamless-m4t-v2-large}"
OUTPUT_DIR="${OUTPUT_DIR:-models/releases/a100}"
REPO_PREFIX="${REPO_PREFIX:-armenian-video-dubbing}"
VISIBILITY_FLAG="${VISIBILITY_FLAG:-}"

python3 scripts/training/export_models.py \
  --models asr tts translation \
  --asr-model "$ASR_MODEL" \
  --tts-model "$TTS_MODEL" \
  --translation-model "$TRANSLATION_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --profile configs/profiles/lightning_a100_80gb_full.yaml \
  --push-to-hub \
  --hf-namespace "$HF_NAMESPACE" \
  --repo-prefix "$REPO_PREFIX" \
  $VISIBILITY_FLAG