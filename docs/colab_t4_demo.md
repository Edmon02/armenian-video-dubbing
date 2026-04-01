# Colab T4 Trial Profile

This guide is the pragmatic path for trying the repository on a free or low-cost Google Colab T4 GPU with a single short video.

## What Changed

The build prompt targets a full production stack: Whisper large-v3, SeamlessM4T v2, Fish-Speech, MuseTalk, Demucs, Docker, API, UI, and training workflows. The current codebase is structurally close to that design, but not every piece is equally mature for Colab:

- ASR inference is usable.
- Translation inference is usable.
- TTS is usable through `edge-tts`; Fish-Speech integration is still partial.
- Lip-sync is optional and too memory-heavy for a reliable first pass on a T4.
- Background separation with Demucs is also expensive for a short Colab trial.
- The training path that is most realistic today is Whisper LoRA smoke testing, not full TTS fine-tuning.

The Colab profile therefore optimizes for one objective: get a short Armenian dubbing run working end-to-end on a T4 with minimal VRAM churn.

## Recommended Trial Scope

- Video length: 8 to 20 seconds
- Resolution: 720p or below
- One visible speaker
- Start with lip-sync disabled
- Start with background preservation disabled
- Use the Colab override: `configs/profiles/colab_t4_demo.yaml`

## Colab Setup

Run these cells in order.

```bash
!git clone --branch feat/colab-t4-demo-profile https://github.com/Edmon02/armenian-video-dubbing.git
%cd armenian-video-dubbing
!apt-get -qq update
!apt-get -qq install -y ffmpeg rubberband-cli
!pip install -q -U pip setuptools wheel
!pip install -q -r requirements-colab.txt
!pip install -q -e . --no-deps
```

Notes:

- The `r2u.stat.illinois.edu` apt warning is a Colab environment quirk and can be ignored if `ffmpeg` and `rubberband-cli` install successfully.
- Colab may also print dependency conflict warnings for preinstalled notebook packages. Those warnings are not the blocking error for this repo.
- The actual blocker fixed in this branch was the editable build backend in `pyproject.toml`.

Optional sanity check:

```bash
!python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## Short-Video Inference

`dubbed_short.mp4` is the output file produced by the pipeline. You do not need to find it in advance.

If you do not already have a short source video, generate a safe synthetic `input_short.mp4` first:

```bash
!python scripts/inference/prepare_demo_video.py \
  --mode generate \
  --output /content/input_short.mp4
```

If you already have your own uploaded video and want to trim it down for Colab:

```bash
!python scripts/inference/prepare_demo_video.py \
  --mode trim \
  --input /content/my_video.mp4 \
  --duration 15 \
  --output /content/input_short.mp4
```

Then use this for the first successful dubbing run:

```bash
!python -m src.pipeline /content/input_short.mp4 \
  --output /content/dubbed_short.mp4 \
  --src-lang eng \
  --dialect eastern \
  --emotion neutral \
  --skip-lipsync \
  --no-background \
  --config-override configs/profiles/colab_t4_demo.yaml
```

If you want the Gradio UI in Colab:

```bash
!python -m src.ui.gradio_app \
  --share \
  --config-override configs/profiles/colab_t4_demo.yaml
```

## Smoke-Test Training

For the training smoke tests below, `input_short.mp4` is not used directly. Those scripts expect dataset manifests and audio-text pairs, not a single demo video clip.

### ASR LoRA Smoke Test

This is the most realistic training test on a T4 in the current repo.

```bash
!python scripts/training/train_asr.py \
  --dataset-type common_voice \
  --cv-dir data/common_voice/manifests \
  --output-dir models/asr/whisper-hy-colab-smoke \
  --max-train-samples 64 \
  --max-eval-samples 16 \
  --config-override configs/profiles/colab_t4_demo.yaml
```

Expected outcome:

- Confirms the training stack runs.
- Produces a tiny adapter/checkpoint.
- Does not produce meaningful production quality.

### TTS Smoke Test

The repository's TTS training script is not yet a full production Fish-Speech training pipeline. Use it only as a preprocessing and plumbing check.

```bash
!python scripts/training/train_tts.py \
  --dataset-type common_voice \
  --cv-dir data/common_voice/manifests \
  --output-dir models/tts/fish-speech-hy-colab-smoke \
  --max-train-samples 16 \
  --config-override configs/profiles/colab_t4_demo.yaml
```

## Why This Profile Works Better On T4

- Uses Whisper `small` instead of the default large profile.
- Keeps SeamlessM4T, but unloads models between stages to reduce VRAM pressure.
- Forces a lightweight TTS backend preference with `edge-tts`.
- Disables MuseTalk and Demucs by default.
- Caps input length at 20 seconds.
- Uses faster video encoding and lower audio sample rate.
- Reduces training batch sizes and LoRA rank for smoke tests.

## First Things To Re-Enable After A Successful Trial

1. Remove `--no-background` and test audio post-processing.
2. Try a 10-second lip-sync sample on Colab Pro or a larger GPU.
3. Move ASR back to `medium` or `large-v3` once memory is stable.
4. Replace `edge-tts` with Fish-Speech after wiring a complete inference/training path.
