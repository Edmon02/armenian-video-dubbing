---
# Copy this file to the model repo as README.md and replace every {{...}}.
# Delete metadata keys that do not apply. Keep `verified: false` on self-reported results.
language:
  - hy
  - en            # only if the model takes or produces English
license: {{apache-2.0 | mit | cc-by-4.0 | other}}
# license_name: {{custom-licence-id}}     # when license: other
# license_link: LICENSE
base_model: {{org/base-model-id}}
base_model_relation: {{finetune | adapter | quantized}}
datasets:
  - {{mozilla-foundation/common_voice_17_0}}
  - {{google/fleurs}}
  - Edmon02/hyvoxpopuli
library_name: {{transformers | peft | nemo | other}}
pipeline_tag: {{automatic-speech-recognition | text-to-speech | translation}}
tags:
  - armenian
  - hy-am
  - eastern-armenian
  - dubbing
  - {{voice-cloning | asr | tts | mt}}
model-index:
  - name: {{repo-name}}
    results:
      - task:
          type: {{automatic-speech-recognition | text-to-speech | translation}}
        dataset:
          name: {{FLEURS hy_am test}}
          type: google/fleurs
          config: hy_am
          split: test
        metrics:
          - name: {{CER}}
            type: {{cer}}
            value: {{0.0}}
            verified: false
        source:
          name: colab_dubbing_ablation.ipynb
          url: https://github.com/Edmon02/armenian-video-dubbing/blob/main/notebooks/colab_dubbing_ablation.ipynb
---

# {{Model name}} (`{{repo-name}}`)

{{One sentence: what it does, for which Armenian variety, and the main use (e.g. "Eastern Armenian voice-cloning TTS for English→Armenian TV dubbing").}}

| | |
|---|---|
| **Task** | {{ASR / TTS / MT / VC}} |
| **Language** | Eastern Armenian (`hy`, `hye`){{, English source}} |
| **Base model** | [{{base}}](https://huggingface.co/{{base}}) ({{base licence}}) |
| **Architecture** | {{class name}} |
| **Input → output** | {{e.g. text + 5–15 s reference clip → 24 kHz mono audio}} |
| **Size / VRAM** | {{params}} · {{fp16 GB}} (T4 {{yes/no}}, 4-bit {{GB}}) |
| **Licence** | {{licence}} · commercial use: **{{yes / no / see below}}** |
| **Status** | {{experimental / beta / production}} |

> **Western Armenian (`hyw`):** {{supported / not supported / untested}}.

## Quick start

```python
{{Minimal, copy-pasteable code that runs on a free Colab T4. Use the exact repo id.}}
```

## Evaluation

All numbers below are from the [ablation notebook](https://github.com/Edmon02/armenian-video-dubbing/blob/main/notebooks/colab_dubbing_ablation.ipynb) at commit `{{git sha}}`, profile `{{free_t4 | paid_l4 | paid_a100}}`. "Baseline" is `{{baseline model}}` on the same clips.

| Metric | What it measures | This model | Baseline | Real recordings |
|---|---|---|---|---|
| {{WER / CER}} | {{ASR accuracy / intelligibility via Armenian back-ASR}} | {{x}} | {{y}} | {{z}} |
| chrF++ / COMET-22 | Translation quality vs FLORES `hye_Armn` | {{x}} | {{y}} | n/a |
| Lines outside 0.85–1.18 stretch | How often audio must be stretched hard to fit timing | {{x%}} | {{y%}} | n/a |
| SECS (WavLM-SV / ECAPA) | Similarity to the original speaker's voice | {{x}} | {{y}} | {{z}} |
| UTMOSv2 | Predicted naturalness (trained on EN/JA — rough indicator for Armenian) | {{x}} | {{y}} | {{z}} |
| VoxLingua107 `p(hy)` / accent leakage | How Armenian it sounds; English-accent leakage index | {{x}} | {{y}} | {{z}} |
| LSE-C / LSE-D | Lip sync (SyncNet) — only for lip-sync models | {{x}} | {{y}} | {{z}} |
| RTF · peak VRAM | Speed and memory on {{GPU}} | {{x}} · {{GB}} | {{y}} | n/a |

**Test data:** {{N}} clips, {{M}} speakers ({{gender split}}), three speaking-rate buckets. FLEURS is read speech, so TV-dialogue quality can be lower than these numbers.

**Human check:** {{N}} raters, native Eastern Armenian speakers, MOS 1–5 and A/B vs baseline: {{results or "not run yet"}}.

## Training

| | |
|---|---|
| **Data** | {{dataset → hours → speakers → licence}} |
| **Filtering** | {{e.g. CER < 10% vs transcript, SNR > 20 dB, 2–15 s clips}} |
| **Text normalisation** | {{numbers → words, Armenian punctuation (։ ՞ ՛ ՜), ligature և}} |
| **Recipe** | {{LoRA r / full FT, lr, steps, batch, precision}} |
| **Hardware** | {{GPU × hours}} |
| **Code** | [{{script}}](https://github.com/Edmon02/armenian-video-dubbing/blob/main/{{path}}) |

## Intended use

- {{English → Eastern Armenian dubbing of content you hold rights to.}}
- {{Research on Armenian speech.}}

## Out of scope and limitations

- Cloning a voice without the speaker's documented consent, impersonation, or fraud.
- {{Western Armenian, singing, overlapping speech, very noisy audio — state what was tested.}}
- {{Known failure modes, e.g. English loanwords, numbers, names, accent carried over from an English reference clip.}}

## Licence and commercial use

| Component | Licence | Commercial |
|---|---|---|
| Base model `{{base}}` | {{licence}} | {{yes/no}} |
| {{Dataset 1}} | {{CC0 / CC-BY-4.0}} | {{yes, attribution}} |
| This model's weights | {{licence}} | {{yes/no}} |

{{If any part is non-commercial, say it here in one sentence.}}

## Ethics and consent

- Voice-cloning inputs require recorded speaker consent (`voice_consent=True` in the pipeline; the request is logged).
- Dubbed outputs from the reference pipeline carry a visible "AI-Dubbed" watermark.
- Report misuse via the [Community tab](https://huggingface.co/{{repo-id}}/discussions).

## Citation

```bibtex
@misc{sahakyan{{year}}{{shortname}},
  author = {Edmon Sahakyan},
  title  = {{{Model name}}},
  year   = {{{year}}},
  url    = {https://huggingface.co/{{repo-id}}}
}
```
