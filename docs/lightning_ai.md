# Lightning.ai Setup Guide

This guide translates the repository's existing Colab/T4 trial path into a practical Lightning.ai workflow.

The short version:

- Use the existing low-VRAM profile: `configs/profiles/colab_t4_demo.yaml`
- Start with a short end-to-end inference run
- Use tiny Common Voice for ASR smoke training
- Use FLEURS Armenian for a cleaner held-out benchmark
- Treat full lip-sync, Demucs-heavy audio restoration, and serious Fish-Speech training as larger-GPU tasks

If you are using a single `A100 80 GB`, the advanced path in this guide now has a dedicated profile: `configs/profiles/lightning_a100_80gb_full.yaml`.

## What The Docs Say After Review

Across the repository docs, three facts matter most for Lightning.ai:

1. The codebase is designed for a full dubbing stack, but the most reliable low-resource path today is the one described in [colab_t4_demo.md](colab_t4_demo.md).
2. The code already supports config overrides end-to-end, so a lightweight Studio run does not require code changes.
3. The ASR smoke path is the most realistic training task on a free or low-cost single GPU. TTS training is still better treated as a plumbing check than a production-quality training path.

## Best GPU Choice By Goal

Use this decision table instead of choosing the largest GPU blindly.

| Goal | Best Choice | Why |
|------|-------------|-----|
| First successful run | `T4` | Cheapest path, enough for the repo's low-VRAM profile |
| Better stability for short demos | `L4` | More VRAM than T4, still practical for trial work |
| Advanced end-to-end tests with fewer compromises | `A100 40 GB`, `A100 80 GB`, or `L40S` | Enough headroom for larger models and fewer unload/reload cycles |
| Serious lip-sync + heavier post-processing + bigger batches | `H100`, `H200`, `RTXP 6000` | This is where the full-stack ambition becomes more realistic |

### Best Single-GPU Choice For This Repo

If you have already chosen `1x A100 80 GB`, use that as the main machine for advanced validation. It is a strong single-GPU option in Lightning for this repository and is enough for the best practical one-box path here.

### Recommended Starting Point

- `T4` if your goal is only to prove the pipeline can run
- `L4` if your goal is to test with fewer out-of-memory failures

For Lightning free tier work, `L4` is the best balance if it is available within your credits. For pure smoke testing, `T4` is enough.

## What Is Realistic On Lightning Free Tier

Based on the repository docs and Lightning's current Studio/free-tier limits, this is the realistic boundary:

### Reliable

- One short video, about 8 to 20 seconds
- ASR + translation + lightweight TTS
- Lip-sync disabled for the first run
- Background separation disabled for the first run
- Tiny Common Voice ASR smoke training
- FLEURS held-out evaluation download and testing

### Risky But Possible On L4 Or Better

- 20 to 45 second videos
- Whisper model size increases beyond the demo profile
- Light post-processing re-enabled
- Slightly larger ASR training subsets

### Best Practical Single-GPU A100 80GB Scope

- 1 to 3 minute clips for advanced testing
- Whisper `large-v3`
- SeamlessM4T with less aggressive constraints
- Fish-Speech as the preferred TTS backend if your environment is wired correctly
- Demucs re-enabled
- MuseTalk re-enabled for selected clips
- Better export and publishing workflow for trained artifacts

The H100 path remains useful if you later want even more headroom, but the A100 80GB path is the right target for your current setup.

### Not A Good First Free-Tier Target

- Full MuseTalk lip-sync as the default path
- Demucs-heavy runs on long videos
- Full Fish-Speech fine-tuning for quality
- Multi-hour dataset crawling and preprocessing inside a single free Studio session

## Best Data For Advanced Testing

Do not use one dataset for every purpose. The cleanest testing plan is staged.

### 1. End-to-End Demo Input

Use one short English source video with:

- one visible speaker
- little background noise
- 720p or lower
- 8 to 20 seconds duration

This is the fastest way to validate the full pipeline behavior.

### 2. ASR Seed Training

Use Mozilla Common Voice Armenian as the first training source.

Why:

- already supported by the repository
- easy to download in tiny slices
- clean enough for LoRA smoke tests
- low operational complexity on a single GPU

Existing helper:

```bash
python scripts/data_collection/download_cv_tiny.py \
  --output-dir data/common_voice \
  --max-train 80 \
  --max-val 20
```

### 3. Clean Held-Out ASR Benchmark

Use FLEURS Armenian for evaluation.

Why:

- cleaner benchmark than a self-assembled YouTube slice
- fixed train/validation/test splits
- useful for repeatable comparisons between runs
- small enough to fit a Lightning test workflow

Repository helper added for this workflow:

```bash
python scripts/data_collection/download_fleurs_eval.py \
  --output-dir data/fleurs_hy \
  --lang-config hy_am
```

### 4. Translation Benchmark

Use FLORES-200 English-Armenian text pairs for translation evaluation.

Why:

- strong multilingual MT benchmark
- aligned English and Armenian sentences
- suitable for translation quality checks without needing speech data

Repository helper added for this workflow:

```bash
python3 scripts/data_collection/download_flores_eval.py \
  --output-dir data/flores_hye \
  --pair-config eng_Latn-hye_Armn
```

### 5. Scale-Up Data For Better ASR

After the smoke path works, scale ASR with the repository's YouTube crawl pipeline.

Why:

- much more domain diversity than Common Voice
- better match for real online video dubbing
- useful after you validate the basic training loop on cleaner data first

### 6. Best Data For TTS

For TTS quality, the best data is not Common Voice.

Use:

- consented single-speaker Armenian studio recordings
- consistent microphone and room conditions
- 2 to 15 second clips
- emotion or speaking-style labels when possible

Use Common Voice only for TTS plumbing checks, not for your best voice-cloning result.

### 7. Best Data For Lip-Sync

Use:

- HDTF-style talking-head video
- your own Armenian talking-head clips with a stable frontal face
- clear mouth visibility and limited scene cuts

This part is better postponed until ASR, translation, and TTS are stable.

## Lightning.ai Step By Step

## 1. Create The Studio

In Lightning.ai:

1. Create a new Studio or notebook workspace.
2. Start on CPU first so you do setup without burning GPU credits.
3. Keep the default persistent storage attached.
4. Open a terminal inside the Studio.

Why CPU first:

- package install and repo clone do not need GPU
- free-tier GPU sessions have restart limits
- Lightning lets you switch from CPU to GPU without rebuilding the workspace

## 2. Clone The Project

```bash
git clone https://github.com/Edmon02/armenian-video-dubbing.git
cd armenian-video-dubbing
```

If you are using your own fork, clone that instead.

## 3. Install System Dependencies

Lightning Studios expose terminal access and current plans advertise sudo access. Install only the small set this repo needs for the low-resource path.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg rubberband-cli libsndfile1
```

## 4. Install Python Dependencies

For Lightning testing, use the lighter dependency set that the repository already prepared for Colab/T4.

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements-colab.txt
python3 -m pip install -e . --no-deps
```

Why not start with the full production environment:

- it is slower to provision
- it increases the chance of dependency issues
- it does not buy you anything for the first Lightning trial

For an `A100 80 GB` Studio, this tradeoff changes. After the first environment sanity check, install the full project dependencies or your training extras before you start long runs.

You can also bootstrap the Studio with one command from the repository root:

```bash
bash scripts/deployment/setup_lightning_a100.sh
```

At minimum, make sure Hugging Face publishing support is available in the environment you use for release packaging:

```bash
python3 -m pip install huggingface_hub
```

## 5. Add Secrets

Set your Hugging Face token before downloading models or gated assets.

```bash
export HF_TOKEN=your_huggingface_token
```

Optional:

```bash
export WANDB_API_KEY=your_wandb_key
export WANDB_PROJECT=armenian-video-dubbing
```

## 6. Verify The Environment

Run a minimal check before requesting a GPU.

```bash
python3 scripts/verify_setup.py
```

Then switch the Studio machine to `T4` or `L4`.

After the GPU starts:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## 7. Prepare A Short Demo Video

If you do not already have a test clip:

```bash
python3 scripts/inference/prepare_demo_video.py \
  --mode generate \
  --output outputs/temp/input_short.mp4
```

If you already uploaded a longer video:

```bash
python3 scripts/inference/prepare_demo_video.py \
  --mode trim \
  --input path/to/your_video.mp4 \
  --duration 15 \
  --output outputs/temp/input_short.mp4
```

## 8. Run The First End-To-End Test

Use the existing low-resource profile exactly as intended.

```bash
python3 -m src.pipeline outputs/temp/input_short.mp4 \
  --output outputs/video/dubbed_short.mp4 \
  --src-lang eng \
  --dialect eastern \
  --emotion neutral \
  --skip-lipsync \
  --no-background \
  --config-override configs/profiles/colab_t4_demo.yaml
```

This is the most important first milestone. Do not enable lip-sync or background preservation until this succeeds.

If you are on `L4` and want a slightly less constrained profile after the first success:

```bash
python3 -m src.pipeline outputs/temp/input_short.mp4 \
  --output outputs/video/dubbed_short_l4.mp4 \
  --src-lang eng \
  --dialect eastern \
  --emotion neutral \
  --skip-lipsync \
  --no-background \
  --config-override configs/profiles/lightning_l4_demo.yaml
```

If you are on `A100 80 GB` and want the best single-GPU path for your current setup:

```bash
python3 -m src.pipeline outputs/temp/input_short.mp4 \
  --output outputs/video/dubbed_short_a100.mp4 \
  --src-lang eng \
  --dialect eastern \
  --emotion neutral \
  --config-override configs/profiles/lightning_a100_80gb_full.yaml
```

For A100 bring-up, still start with a short clip. After that succeeds, move to longer clips and selectively test lip-sync and Demucs-heavy runs.

## 9. Download Training And Evaluation Data

Tiny Common Voice for ASR smoke training:

```bash
python3 scripts/data_collection/download_cv_tiny.py \
  --output-dir data/common_voice \
  --max-train 80 \
  --max-val 20
```

FLEURS Armenian for cleaner evaluation:

```bash
python3 scripts/data_collection/download_fleurs_eval.py \
  --output-dir data/fleurs_hy \
  --lang-config hy_am \
  --max-test 100
```

FLORES English→Armenian for translation evaluation:

```bash
python3 scripts/data_collection/download_flores_eval.py \
  --output-dir data/flores_hye \
  --pair-config eng_Latn-hye_Armn \
  --max-devtest 200
```

## 10. Run The ASR Smoke Train

```bash
python3 scripts/training/train_asr.py \
  --dataset-type common_voice \
  --cv-dir data/common_voice/manifests \
  --output-dir models/asr/whisper-hy-lightning-smoke \
  --max-train-samples 64 \
  --max-eval-samples 16 \
  --config-override configs/profiles/colab_t4_demo.yaml
```

Expected result:

- confirms the training stack runs in Lightning
- produces a small adapter/checkpoint
- does not yet prove production quality

For a stronger A100 run, increase sample counts and use the A100 profile:

```bash
python3 scripts/training/train_asr.py \
  --dataset-type common_voice \
  --cv-dir data/common_voice/manifests \
  --output-dir models/asr/whisper-hy-a100 \
  --max-train-samples 2000 \
  --max-eval-samples 200 \
  --config-override configs/profiles/lightning_a100_80gb_full.yaml
```

## 11. Optional TTS Smoke Test

Use this only after ASR smoke training works.

```bash
python3 scripts/training/train_tts.py \
  --dataset-type common_voice \
  --cv-dir data/common_voice/manifests \
  --output-dir models/tts/fish-speech-hy-lightning-smoke \
  --max-train-samples 16 \
  --config-override configs/profiles/colab_t4_demo.yaml
```

Interpret this as a pipeline test, not a final TTS-quality benchmark.

For an A100-quality pass, prefer consented studio speech rather than Common Voice and run with the A100 profile.

## 11.5 Package And Push Models To Hugging Face

Use the export utility to stage local model folders into release bundles and publish them to your Hugging Face account.

Environment:

```bash
export HF_TOKEN=your_huggingface_token
```

Example upload for ASR, TTS, and translation artifacts:

```bash
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
```

This creates or reuses these model repositories:

- `YOUR_HF_USERNAME/armenian-video-dubbing-asr`
- `YOUR_HF_USERNAME/armenian-video-dubbing-tts`
- `YOUR_HF_USERNAME/armenian-video-dubbing-translation`

If you want public repos, remove `--private`.

For repeated A100 releases, use the wrapper script instead of retyping the full command:

```bash
export HF_NAMESPACE=YOUR_HF_USERNAME
export HF_TOKEN=your_huggingface_token
export VISIBILITY_FLAG=
bash scripts/training/publish_a100_release.sh
```

If you want private repos through the wrapper, set:

```bash
export VISIBILITY_FLAG=--private
```

## 12. Evaluate Before Scaling Up

Run the repository evaluation entry point after you have a first model artifact or demo output.

```bash
python3 scripts/evaluation/evaluate_full.py
```

For translation-specific checks:

```bash
python3 scripts/training/evaluate_translation.py \
  --test-data data/flores_hye/manifests/combined.jsonl \
  --output-dir outputs/translation_eval
```

## Upgrade Path After The First Success

### Move From T4 To L4

Do this when:

- the demo pipeline works but memory is tight
- you want slightly longer clips
- you want to try less aggressive settings

### Move From L4 To A100 Or Better

Do this when:

- you want to re-enable Demucs and experiment with lip-sync
- you want larger ASR training subsets
- you want to test more realistic end-to-end quality

### Move From A100 To H100 Full Path

Do this when:

- you want the best single-GPU path for this repository
- you want to test Demucs and MuseTalk together on selected samples
- you want to package and publish larger model bundles with less compromise

## Practical Lightning Tips

- Use CPU for cloning, installs, and dataset manifest prep whenever possible.
- Save outputs and checkpoints under the repository workspace so they remain in persistent storage.
- Free-tier Studios currently have restart limits, so checkpoint often.
- Keep your first video short enough that one failed run does not waste most of a session.
- If a T4 keeps failing, move to L4 before changing many parameters at once.
- On A100, do not jump straight to multi-minute clips with every heavy feature on. Prove the short clip first, then scale one dimension at a time.

## Recommended Test Plan

If you want the highest signal with the fewest wasted credits, use this order:

1. Short inference run on `T4`
2. Tiny Common Voice ASR smoke training on `T4`
3. FLEURS Armenian evaluation download and benchmark
4. Repeat on `L4` with a slightly less constrained trial
5. Only then try lip-sync, heavier post-processing, or larger training subsets

This gives you a clean progression from proof-of-life to meaningful testing.