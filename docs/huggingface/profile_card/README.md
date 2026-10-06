---
title: README
emoji: 🎙️
colorFrom: red
colorTo: yellow
sdk: static
pinned: false
---

# Edmon Sahakyan · Armenian speech & dubbing

I build open models for **Eastern Armenian (`hy`) speech**: recognition, synthesis, translation, and end-to-end **English → Armenian video dubbing** for TV and drama.

Every card on this profile reports how the numbers were measured, and checkpoints that are still experimental are labelled as experimental.

## Current models

| Model | Task | Status |
|---|---|---|
| [speecht5_finetuned_voxpopuli_hy](https://huggingface.co/Edmon02/speecht5_finetuned_voxpopuli_hy) | Armenian TTS (SpeechT5, 2 speakers) | Main TTS checkpoint |
| [TTS_NB_2](https://huggingface.co/Edmon02/TTS_NB_2) · [TTS_NB_ONNX](https://huggingface.co/Edmon02/TTS_NB_ONNX) | Armenian TTS for training / ONNX | Active |
| [speecht5_finetuned_hy](https://huggingface.co/Edmon02/speecht5_finetuned_hy) | Armenian TTS (Common Voice) | Earlier checkpoint |
| [whisper-small-hy](https://huggingface.co/Edmon02/whisper-small-hy) | Armenian ASR | Experimental (validation WER ≈ 75%) |
| [marian-finetuned-kde4-en-to-hy](https://huggingface.co/Edmon02/marian-finetuned-kde4-en-to-hy) | English → Armenian MT (software domain) | Experimental |

**Data:** [HyVoxPopuli](https://huggingface.co/datasets/Edmon02/hyvoxpopuli) (Armenian speech) · **Demo:** [SpeechT5 Armenian TTS](https://huggingface.co/spaces/Edmon02/SpeechT5_hy)

## Coming next: Armenian dubbing stack

From the [armenian-video-dubbing](https://github.com/Edmon02/armenian-video-dubbing) project. Release names and dates are not fixed yet.

| Planned release | What it does | Licence target |
|---|---|---|
| Armenian voice-cloning TTS | Fine-tune of an open multilingual TTS on Common Voice `hy-AM` (CC0) and FLEURS `hy_am` (CC-BY-4.0) | Commercial-friendly |
| Duration-aware EN → HY translation | Fits each Armenian line to the original timing (syllable budget) | Commercial-friendly |
| Armenian ASR adapter | Whisper LoRA for Armenian back-transcription and QA | Apache-2.0 / MIT base |
| Dubbing evaluation set | Public multi-speaker clips plus scoring notebook (WER, chrF++, COMET, SECS, accent leakage, LSE-C/D) | Per-source licences |

Each release will ship with a reproducible score from the [ablation notebook](https://github.com/Edmon02/armenian-video-dubbing/blob/main/notebooks/colab_dubbing_ablation.ipynb), a licence statement covering every training source, and voice-cloning consent rules.

## Principles

- **Measured, not claimed.** Metrics come from public test sets with a script anyone can rerun.
- **Licences are explicit.** Non-commercial components are marked and never used in a commercial release.
- **Consent first.** Voice cloning is only for speakers who agreed to it, and dubbed output is watermarked.

## Contact

[GitHub](https://github.com/Edmon02) · [LinkedIn](https://www.linkedin.com/in/edmon-sahakyan-64798619a) · [Bluesky](https://bsky.app/profile/edmon02.bsky.social) · Open to collaboration on Armenian speech and language data.
