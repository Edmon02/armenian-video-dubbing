#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

info "Lightning A100 bootstrap starting"
info "Project root: $PROJECT_ROOT"

command -v python3 >/dev/null 2>&1 || err "python3 is required"
command -v git >/dev/null 2>&1 || err "git is required"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
    ok "Detected GPU: $GPU_NAME ($GPU_MEM)"
    if [[ "$GPU_NAME" != *"A100"* ]]; then
        warn "This script is optimized for A100, but current GPU is: $GPU_NAME"
    fi
else
    warn "No NVIDIA GPU detected yet. Continue if you are still on CPU setup stage."
fi

if command -v sudo >/dev/null 2>&1; then
    info "Installing system packages"
    sudo apt-get update
    sudo apt-get install -y ffmpeg rubberband-cli libsndfile1 git-lfs
else
    warn "sudo not available; skipping apt packages"
fi

info "Upgrading pip tooling"
python3 -m pip install --upgrade pip setuptools wheel

info "Installing Python dependencies"
python3 -m pip install -r requirements-colab.txt
python3 -m pip install -e . --no-deps
python3 -m pip install huggingface_hub

info "Running repository verification"
python3 scripts/verify_setup.py || warn "verify_setup.py reported issues; inspect output before long runs"

cat <<'EOF'

Next recommended commands on Lightning A100 80GB:

1. Export your token:
   export HF_TOKEN=your_huggingface_token

2. First short end-to-end test:
   python3 -m src.pipeline outputs/temp/input_short.mp4 \
     --output outputs/video/dubbed_short_a100.mp4 \
     --src-lang eng \
     --dialect eastern \
     --emotion neutral \
     --config-override configs/profiles/lightning_a100_80gb_full.yaml

3. Stronger ASR run:
   python3 scripts/training/train_asr.py \
     --dataset-type common_voice \
     --cv-dir data/common_voice/manifests \
     --output-dir models/asr/whisper-hy-a100 \
     --max-train-samples 2000 \
     --max-eval-samples 200 \
     --config-override configs/profiles/lightning_a100_80gb_full.yaml

4. Package and push models to Hugging Face:
   python3 scripts/training/export_models.py \
     --models asr tts translation \
     --asr-model models/asr/whisper-hy-a100 \
     --tts-model models/tts/fish-speech-hy-lightning-smoke \
     --translation-model models/translation/seamless-m4t-v2-large \
     --output-dir models/releases/a100 \
     --profile configs/profiles/lightning_a100_80gb_full.yaml \
     --push-to-hub \
     --hf-namespace YOUR_HF_USERNAME \
     --repo-prefix armenian-video-dubbing \
     --private

EOF

ok "Lightning A100 bootstrap complete"