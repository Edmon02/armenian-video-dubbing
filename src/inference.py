#!/usr/bin/env python3
"""
Core Inference Modules — Phase 3

Production wrappers around fine-tuned models:
  1. ASRInference        — Whisper large-v3 + LoRA (segment-level w/ timestamps)
  2. TranslationInference — SeamlessM4T v2 Large (text-to-text)
  3. TTSInference         — Fish-Speech S2 Pro / edge-tts fallback
  4. LipSyncInference     — MuseTalk v1.5+ real-time mouth inpainting
  5. AudioPostProcessor   — Demucs source sep, denoise, normalize, mix
"""

import gc
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
from loguru import logger


def _resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    """Resolve a string dtype name into a torch dtype compatible with the device."""
    if device == "cpu":
        return torch.float32

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_name.lower(), torch.float16)


# ============================================================================
# ASR Inference
# ============================================================================

class ASRInference:
    """ASR inference using Whisper large-v3 with optional LoRA adapter.

    Returns segment-level transcriptions with word-level timestamps so
    downstream translation + TTS can be aligned per-segment.
    """

    def __init__(
        self,
        model_path: Path = Path("models/asr/whisper-hy-full"),
        model_id: str = "openai/whisper-large-v3",
        device: str = "cuda",
        use_fp16: bool = True,
        quantize_bits: int = 0,
        language: str = "hy",
        task: str = "transcribe",
        chunk_length_s: int = 30,
        batch_size: int = 8,
    ):
        self.model_path = Path(model_path)
        self.model_id = model_id
        self.device = device
        self.use_fp16 = use_fp16
        self.quantize_bits = quantize_bits  # 0=off, 4=int4, 8=int8
        self.language = language
        self.task = task
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self._pipe = None  # faster-whisper / HF pipeline

    def load(self):
        """Load model + optional LoRA adapter."""
        if self.model is not None:
            return

        logger.info("Loading ASR model from {}...", self.model_path)

        try:
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
                pipeline as hf_pipeline,
            )

            base_model_id = self.model_id
            use_accelerate = False

            # Build quantization config if requested
            quant_config = None
            if self.quantize_bits in (4, 8):
                try:
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=(self.quantize_bits == 4),
                        load_in_8bit=(self.quantize_bits == 8),
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                    use_accelerate = True
                    logger.info("Using {}-bit quantization", self.quantize_bits)
                except ImportError:
                    logger.warning("bitsandbytes not installed; falling back to fp16")

            model_kwargs = dict(
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                low_cpu_mem_usage=True,
            )
            if quant_config:
                model_kwargs["quantization_config"] = quant_config
                model_kwargs["device_map"] = "auto"

            pipeline_kwargs = {
                "task": "automatic-speech-recognition",
                "torch_dtype": torch.float16 if self.use_fp16 else torch.float32,
            }
            if not use_accelerate:
                if self.device == "cuda":
                    pipeline_kwargs["device"] = 0
                elif self.device == "cpu":
                    pipeline_kwargs["device"] = -1

            # Try loading fine-tuned checkpoint first, fall back to base
            model_id = base_model_id
            if self.model_path.exists() and (self.model_path / "config.json").exists():
                model_id = str(self.model_path)
                logger.info("Loading fine-tuned model from {}", model_id)
            elif self.model_path.exists() and (self.model_path / "adapter_config.json").exists():
                # LoRA adapter on top of base
                model = WhisperForConditionalGeneration.from_pretrained(
                    base_model_id,
                    **model_kwargs,
                )
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(self.model_path))
                model = model.merge_and_unload()
                if not use_accelerate and self.device != "cpu":
                    model = model.to(self.device)
                model.eval()
                self.model = model
                self.processor = WhisperProcessor.from_pretrained(base_model_id)
                self._pipe = hf_pipeline(
                    model=self.model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor,
                    **pipeline_kwargs,
                )
                logger.info("ASR model loaded with LoRA adapter")
                return
            else:
                logger.info("Fine-tuned model not found, using base {}", base_model_id)

            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                **model_kwargs,
            )
            if not use_accelerate and self.device != "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()

            self._pipe = hf_pipeline(
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                **pipeline_kwargs,
            )

            logger.info("ASR model loaded ({})", model_id)

        except Exception as e:
            logger.error("Failed to load ASR model: {}", e)
            raise

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file and return segment-level results with timestamps."""
        if self.model is None:
            self.load()

        try:
            result = self._pipe(
                audio_path,
                generate_kwargs={
                    "language": "armenian" if self.language in {"hy", "hye", "armenian"} else self.language,
                    "task": self.task,
                },
                return_timestamps=True,
                chunk_length_s=self.chunk_length_s,
                batch_size=self.batch_size,
            )

            # Build segments from chunks
            segments = []
            if "chunks" in result:
                for chunk in result["chunks"]:
                    ts = chunk.get("timestamp", (0.0, None))
                    segments.append({
                        "text": chunk["text"].strip(),
                        "start": ts[0] if ts[0] is not None else 0.0,
                        "end": ts[1] if ts[1] is not None else 0.0,
                    })
            else:
                # Single-chunk fallback
                segments.append({
                    "text": result["text"].strip(),
                    "start": 0.0,
                    "end": 0.0,
                })

            import librosa
            duration = librosa.get_duration(path=audio_path)

            return {
                "text": result["text"].strip(),
                "language": "hy",
                "segments": segments,
                "duration_sec": duration,
            }

        except Exception as e:
            logger.error("ASR transcription failed: {}", e)
            return {"text": "", "segments": [], "error": str(e)}

    def batch_transcribe(self, audio_list: list) -> list:
        """Transcribe a list of audio arrays. Returns list of text strings."""
        if self.model is None:
            self.load()

        results = []
        for audio in audio_list:
            # Save temp file for pipeline
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import soundfile as sf
                sf.write(f.name, audio, 16000)
                r = self.transcribe(f.name)
                results.append(r.get("text", ""))
                os.unlink(f.name)
        return results

    def free_memory(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self._pipe
            self.model = None
            self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# Translation Inference
# ============================================================================

class TranslationInference:
    """Text-to-text translation using SeamlessM4T v2 Large.

    Supports eng→hye (English to Eastern Armenian) and other language pairs
    covered by SeamlessM4T.
    """

    def __init__(
        self,
        model_id: str = "facebook/seamless-m4t-v2-large",
        device: str = "cuda",
        dtype: str = "float16",
        num_beams: int = 5,
        max_new_tokens: int = 512,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self):
        """Load SeamlessM4T v2 model for text-to-text translation."""
        if self.model is not None:
            return

        logger.info("Loading translation model {}...", self.model_id)

        try:
            from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

            torch_dtype = _resolve_torch_dtype(self.dtype, self.device)

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = SeamlessM4Tv2ForTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()

            logger.info("Translation model loaded: {}", self.model_id)

        except Exception as e:
            logger.error("Failed to load translation model: {}", e)
            raise

    def translate(
        self,
        text: str,
        src_lang: str = "eng",
        tgt_lang: str = "hye",
    ) -> dict:
        """Translate text from source to target language.

        Args:
            text: Source text to translate.
            src_lang: Source language code (SeamlessM4T format).
            tgt_lang: Target language code (SeamlessM4T format).

        Returns:
            Dict with src_text, tgt_text, src_lang, tgt_lang.
        """
        if not text or not text.strip():
            return {"src_text": text, "tgt_text": "", "src_lang": src_lang, "tgt_lang": tgt_lang}

        if self.model is None:
            self.load()

        try:
            inputs = self.processor(
                text=text,
                src_lang=src_lang,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    tgt_lang=tgt_lang,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                )

            translated_text = self.processor.batch_decode(
                output_tokens, skip_special_tokens=True,
            )[0]

            return {
                "src_text": text,
                "tgt_text": translated_text,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }

        except Exception as e:
            logger.error("Translation failed: {}", e)
            return {"src_text": text, "tgt_text": text, "error": str(e)}

    def translate_segments(
        self,
        segments: List[Dict],
        src_lang: str = "eng",
        tgt_lang: str = "hye",
    ) -> List[Dict]:
        """Translate a list of timed segments, preserving timestamps.

        Args:
            segments: List of dicts with keys: text, start, end
            src_lang: Source language
            tgt_lang: Target language

        Returns:
            List of dicts with translated text + original timestamps.
        """
        translated_segments = []
        for seg in segments:
            result = self.translate(seg["text"], src_lang, tgt_lang)
            translated_segments.append({
                "text": result["tgt_text"],
                "src_text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
            })
        return translated_segments

    def free_memory(self):
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# TTS Inference
# ============================================================================

class TTSInference:
    """TTS inference with multiple backends:

    1. Fish-Speech S2 Pro (primary — if externals/fish-speech is available)
    2. edge-tts (fallback — free Microsoft TTS via edge_tts package)

    Both support voice cloning / voice selection. Fish-Speech does zero-shot
    cloning from a reference audio clip. edge-tts uses predefined voices.
    """

    def __init__(
        self,
        model_path: Path = Path("models/tts/fish-speech-hy"),
        device: str = "cuda",
        sample_rate: int = 44100,
        preferred_backend: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.sample_rate = sample_rate
        self.preferred_backend = preferred_backend
        self.backend = None  # "fish-speech" or "edge-tts"
        self.model = None
        self.encoder = None

    def load(self):
        """Detect and load best available TTS backend."""
        if self.backend is not None:
            return

        backend_order = []
        if self.preferred_backend and self.preferred_backend != "auto":
            backend_order.append(self.preferred_backend)
        backend_order.extend(
            backend for backend in ["fish-speech", "edge-tts", "gtts"]
            if backend not in backend_order
        )

        fish_dir = Path("externals/fish-speech")
        for backend in backend_order:
            if backend == "fish-speech":
                if fish_dir.exists() and (fish_dir / "fish_speech").exists():
                    try:
                        self._load_fish_speech()
                        self.backend = "fish-speech"
                        logger.info("TTS backend: Fish-Speech S2 Pro")
                        return
                    except Exception as e:
                        logger.warning("Fish-Speech failed to load: {}", e)

            if backend == "edge-tts":
                try:
                    import edge_tts  # noqa: F401
                    self.backend = "edge-tts"
                    logger.info("TTS backend: edge-tts (Microsoft)")
                    return
                except ImportError:
                    continue

            if backend == "gtts":
                try:
                    import gtts  # noqa: F401
                    self.backend = "gtts"
                    logger.info("TTS backend: gTTS (Google)")
                    return
                except ImportError:
                    continue

        raise RuntimeError(
            "No TTS backend available. Install one of: "
            "fish-speech (externals/), edge-tts (pip install edge-tts), "
            "or gtts (pip install gtts)"
        )

    def _load_fish_speech(self):
        """Load Fish-Speech model."""
        import sys
        fish_dir = Path("externals/fish-speech")
        if str(fish_dir) not in sys.path:
            sys.path.insert(0, str(fish_dir))

        from fish_speech.models.vqgan.lit_module import VQGAN
        from fish_speech.models.text2semantic.llama import TextToSemantic

        # Load from checkpoint or HuggingFace
        logger.info("Loading Fish-Speech from {}", self.model_path)
        # The actual loading depends on fish-speech version; this is the pattern
        self.model = {"loaded": True}  # Placeholder until fish-speech API stabilizes

    def synthesize(
        self,
        text: str,
        reference_audio: Optional[np.ndarray] = None,
        reference_audio_path: Optional[str] = None,
        emotion: str = "neutral",
        speaker_id: Optional[int] = None,
        language: str = "hy",
    ) -> dict:
        """Synthesize speech from text.

        Args:
            text: Text to speak.
            reference_audio: Reference audio array for voice cloning (Fish-Speech).
            reference_audio_path: Path to reference audio file.
            emotion: Emotion tag (neutral, happy, sad, angry, excited, calm).
            speaker_id: Optional speaker ID.
            language: Language code.

        Returns:
            Dict with audio (np.ndarray), sample_rate, duration_sec.
        """
        if self.backend is None:
            self.load()

        if not text or not text.strip():
            return {
                "audio": np.zeros(int(0.5 * self.sample_rate), dtype=np.float32),
                "sample_rate": self.sample_rate,
                "duration_sec": 0.5,
                "text": text,
                "emotion": emotion,
            }

        if self.backend == "fish-speech":
            return self._synthesize_fish_speech(text, reference_audio_path, emotion, language)
        elif self.backend == "edge-tts":
            try:
                return self._synthesize_edge_tts(text, emotion, language)
            except Exception as e:
                logger.warning("edge-tts synthesis failed: {}, falling back to gTTS", e)
                return self._synthesize_gtts(text, language)
        elif self.backend == "gtts":
            return self._synthesize_gtts(text, language)
        else:
            raise RuntimeError(f"Unknown TTS backend: {self.backend}")

    def _synthesize_fish_speech(
        self, text: str, reference_audio_path: Optional[str], emotion: str, language: str
    ) -> dict:
        """Synthesize using Fish-Speech S2 Pro."""
        import sys

        fish_dir = Path("externals/fish-speech")
        if str(fish_dir) not in sys.path:
            sys.path.insert(0, str(fish_dir))

        # Build command for Fish-Speech CLI inference
        cmd = [
            sys.executable, "-m", "fish_speech.inference",
            "--text", text,
            "--output", str(Path(tempfile.mkdtemp()) / "output.wav"),
        ]

        if reference_audio_path:
            cmd.extend(["--reference-audio", str(reference_audio_path)])

        if emotion != "neutral":
            cmd.extend(["--emotion", emotion])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, cwd=str(fish_dir)
            )
            if result.returncode == 0:
                output_path = cmd[cmd.index("--output") + 1]
                import soundfile as sf
                audio, sr = sf.read(output_path)
                if sr != self.sample_rate:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    sr = self.sample_rate
                return {
                    "audio": audio.astype(np.float32),
                    "sample_rate": sr,
                    "duration_sec": len(audio) / sr,
                    "text": text,
                    "emotion": emotion,
                    "backend": "fish-speech",
                }
        except Exception as e:
            logger.warning("Fish-Speech inference failed: {}, falling back to edge-tts", e)

        # Fallback to edge-tts
        return self._synthesize_edge_tts(text, emotion, language)

    def _synthesize_edge_tts(self, text: str, emotion: str, language: str) -> dict:
        """Synthesize using Microsoft edge-tts.

        edge-tts provides high-quality neural voices with no API key required.
        For Armenian, it uses the closest available voice.
        """
        import asyncio

        try:
            import edge_tts
        except ImportError:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

        normalized_language = {
            "hye": "hy",
            "hyw": "hy",
        }.get(language, language)

        voice_candidates = {
            "hy": [
                "hy-AM-AnahitNeural",
                "hy-AM-HaykNeural",
                "en-US-AvaMultilingualNeural",
                "en-US-AndrewMultilingualNeural",
            ],
            "en": ["en-US-JennyNeural", "en-US-GuyNeural"],
            "ru": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
        }
        candidate_voices = voice_candidates.get(normalized_language, voice_candidates.get("hy", []))
        if not candidate_voices:
            candidate_voices = ["en-US-AvaMultilingualNeural"]

        # Map emotion to SSML prosody parameters
        emotion_prosody = {
            "neutral": {"rate": "0%", "pitch": "0%", "volume": "0%"},
            "happy":   {"rate": "+10%", "pitch": "+5Hz", "volume": "+5%"},
            "sad":     {"rate": "-15%", "pitch": "-5Hz", "volume": "-10%"},
            "angry":   {"rate": "+5%", "pitch": "+10Hz", "volume": "+15%"},
            "excited": {"rate": "+15%", "pitch": "+10Hz", "volume": "+10%"},
            "calm":    {"rate": "-10%", "pitch": "-3Hz", "volume": "-5%"},
        }
        prosody = emotion_prosody.get(emotion, emotion_prosody["neutral"])

        async def _do_tts(voice: str):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_mp3 = f.name

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=prosody["rate"],
                pitch=prosody["pitch"],
                volume=prosody["volume"],
            )
            await communicate.save(tmp_mp3)
            return tmp_mp3

        def _run_tts(voice: str) -> str:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _do_tts(voice))
                        return future.result(timeout=60)
                return asyncio.run(_do_tts(voice))
            except RuntimeError:
                return asyncio.run(_do_tts(voice))

        last_error = None
        selected_voice = candidate_voices[0]
        tmp_mp3 = None
        for voice in candidate_voices:
            try:
                tmp_mp3 = _run_tts(voice)
                selected_voice = voice
                logger.info("edge-tts synthesis succeeded with voice {}", voice)
                break
            except Exception as e:
                last_error = e
                logger.warning("edge-tts voice {} failed: {}", voice, e)

        if tmp_mp3 is None:
            raise RuntimeError(f"edge-tts failed for all candidate voices: {last_error}")

        # Convert mp3 to wav and load
        tmp_wav = tmp_mp3.replace(".mp3", ".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", str(self.sample_rate), "-ac", "1", tmp_wav],
            capture_output=True, check=True,
        )

        import soundfile as sf
        audio, sr = sf.read(tmp_wav)

        # Cleanup temp files
        try:
            os.unlink(tmp_mp3)
            os.unlink(tmp_wav)
        except OSError:
            pass

        return {
            "audio": audio.astype(np.float32),
            "sample_rate": sr,
            "duration_sec": len(audio) / sr,
            "text": text,
            "emotion": emotion,
            "backend": "edge-tts",
            "voice": selected_voice,
        }

    def _synthesize_gtts(self, text: str, language: str) -> dict:
        """Synthesize using gTTS (Google Text-to-Speech) as last resort."""
        from gtts import gTTS

        lang_code = "hy" if language in ("hy", "hye") else language

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_mp3 = f.name

        tts = gTTS(text=text, lang=lang_code)
        tts.save(tmp_mp3)

        tmp_wav = tmp_mp3.replace(".mp3", ".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", str(self.sample_rate), "-ac", "1", tmp_wav],
            capture_output=True, check=True,
        )

        import soundfile as sf
        audio, sr = sf.read(tmp_wav)

        try:
            os.unlink(tmp_mp3)
            os.unlink(tmp_wav)
        except OSError:
            pass

        return {
            "audio": audio.astype(np.float32),
            "sample_rate": sr,
            "duration_sec": len(audio) / sr,
            "text": text,
            "backend": "gtts",
        }

    def extract_speaker_embedding(self, reference_audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from reference audio."""
        try:
            from resemblyzer import VoiceEncoder

            encoder = VoiceEncoder("cpu", verbose=False)

            if len(reference_audio.shape) > 1:
                reference_audio = reference_audio.mean(axis=1)

            embedding = encoder.embed_utterance(reference_audio)
            return embedding

        except Exception as e:
            logger.warning("Speaker embedding extraction failed: {}", e)
            return np.zeros(256)

    def free_memory(self):
        if self.model is not None:
            del self.model
            self.model = None
        self.backend = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# Lip-Sync Inference
# ============================================================================

class LipSyncInference:
    """Lip-sync inference using MuseTalk v1.5+.

    Inpaints mouth regions of video frames to match dubbed audio.
    Falls back gracefully if MuseTalk is not installed (returns original video).
    """

    def __init__(
        self,
        model_path: Path = Path("models/lipsync/MuseTalk"),
        device: str = "cuda",
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.musetalk_dir = Path("externals/MuseTalk")
        self.available = False

    def load(self):
        """Check if MuseTalk is available and load models."""
        if self.available:
            return

        if not self.musetalk_dir.exists():
            logger.warning(
                "MuseTalk not found at {}. Lip-sync will be skipped. "
                "Clone it: git clone https://github.com/TMElyralab/MuseTalk externals/MuseTalk",
                self.musetalk_dir,
            )
            return

        # Verify MuseTalk has required files
        required = ["musetalk", "configs"]
        missing = [r for r in required if not (self.musetalk_dir / r).exists()]
        if missing:
            logger.warning("MuseTalk is incomplete (missing: {}). Lip-sync disabled.", missing)
            return

        self.available = True
        logger.info("MuseTalk loaded from {}", self.musetalk_dir)

    def inpaint_mouth(
        self,
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        fps: int = 25,
        bbox_shift: int = 0,
    ) -> dict:
        """Inpaint mouth movements to match dubbed audio.

        Args:
            video_path: Input video file.
            audio_path: Dubbed audio file to sync to.
            output_path: Where to save the lip-synced video.
            fps: Video frame rate.
            bbox_shift: Face bounding box vertical shift.

        Returns:
            Dict with status, output video path, or error.
        """
        if not self.available:
            self.load()

        if not self.available:
            logger.warning("MuseTalk not available — returning original video")
            return {
                "status": "skipped",
                "video_path": video_path,
                "output": video_path,
                "note": "MuseTalk not installed; lip-sync skipped",
            }

        if output_path is None:
            output_path = str(
                Path("outputs/temp") / f"{Path(video_path).stem}_lipsync.mp4"
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            import sys

            # MuseTalk inference via its CLI
            cmd = [
                sys.executable, "-m", "musetalk.inference",
                "--video_path", str(video_path),
                "--audio_path", str(audio_path),
                "--output_path", str(output_path),
                "--bbox_shift", str(bbox_shift),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.musetalk_dir) + ":" + env.get("PYTHONPATH", "")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(self.musetalk_dir),
                env=env,
            )

            if result.returncode == 0 and Path(output_path).exists():
                logger.info("Lip-sync complete: {}", output_path)
                return {
                    "status": "success",
                    "video_path": video_path,
                    "output": output_path,
                    "fps": fps,
                }
            else:
                stderr = result.stderr[:500] if result.stderr else "unknown error"
                logger.warning("MuseTalk failed: {}", stderr)

                # Try alternative MuseTalk CLI patterns
                return self._fallback_lipsync(video_path, audio_path, output_path, fps)

        except subprocess.TimeoutExpired:
            logger.error("MuseTalk timed out after 600s")
            return {"status": "skipped", "output": video_path, "error": "timeout"}
        except Exception as e:
            logger.error("Lip-sync failed: {}", e)
            return {"status": "skipped", "output": video_path, "error": str(e)}

    def _fallback_lipsync(
        self, video_path: str, audio_path: str, output_path: str, fps: int
    ) -> dict:
        """Fallback: try running MuseTalk's realtime inference script."""
        try:
            import sys

            script = self.musetalk_dir / "scripts" / "inference.sh"
            if not script.exists():
                # Try the Python entry point directly
                inference_py = self.musetalk_dir / "musetalk" / "real_time_inference.py"
                if inference_py.exists():
                    cmd = [
                        sys.executable, str(inference_py),
                        "--video_path", str(video_path),
                        "--audio_path", str(audio_path),
                        "--result_dir", str(Path(output_path).parent),
                    ]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=600,
                        cwd=str(self.musetalk_dir),
                    )
                    if result.returncode == 0:
                        return {"status": "success", "output": output_path, "fps": fps}

            logger.warning("All MuseTalk methods failed — skipping lip-sync")
            return {"status": "skipped", "output": video_path, "note": "MuseTalk inference failed"}

        except Exception as e:
            return {"status": "skipped", "output": video_path, "error": str(e)}

    def free_memory(self):
        self.available = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# Audio Post-Processing
# ============================================================================

class AudioPostProcessor:
    """Post-process dubbed audio: source separation, denoise, normalize, mix."""

    def __init__(self, sample_rate: int = 44100, enable_source_separation: bool = True):
        self.sample_rate = sample_rate
        self.enable_source_separation = enable_source_separation
        self.demucs = None

    def load_demucs(self):
        """Load Demucs for source separation."""
        if self.demucs is not None:
            return

        logger.info("Loading Demucs (source separation)...")

        try:
            from demucs.pretrained import get_model
            self.demucs = get_model("htdemucs_ft")
            logger.info("Demucs loaded")
        except Exception as e:
            logger.warning("Demucs not available: {}", e)

    def separate_sources(self, audio: np.ndarray) -> dict:
        """Separate audio into vocals and accompaniment using Demucs.

        Returns dict with keys: vocals, drums, bass, other
        """
        if not self.enable_source_separation:
            return {"vocals": audio, "accompaniment": np.zeros_like(audio)}

        self.load_demucs()
        if self.demucs is None:
            return {"vocals": audio, "accompaniment": np.zeros_like(audio)}

        try:
            from demucs.apply import apply_model

            # Demucs expects (batch, channels, samples)
            if audio.ndim == 1:
                audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
            else:
                audio_tensor = torch.tensor(audio).unsqueeze(0)

            audio_tensor = audio_tensor.float()

            with torch.no_grad():
                sources = apply_model(self.demucs, audio_tensor, device="cpu")

            # sources shape: (batch, n_sources, channels, samples)
            # htdemucs_ft sources: drums, bass, other, vocals
            vocals = sources[0, 3, 0].numpy()  # vocals
            accompaniment = (sources[0, 0, 0] + sources[0, 1, 0] + sources[0, 2, 0]).numpy()

            return {"vocals": vocals, "accompaniment": accompaniment}

        except Exception as e:
            logger.warning("Demucs separation failed: {}", e)
            return {"vocals": audio, "accompaniment": np.zeros_like(audio)}

    def denoise_audio(self, audio: np.ndarray) -> np.ndarray:
        """Noise reduction using spectral gating."""
        # Estimate noise from first 0.5s (assuming relative silence at start)
        noise_frames = min(int(0.5 * self.sample_rate), len(audio) // 4)
        if noise_frames < 100:
            return audio

        noise_profile = np.mean(np.abs(audio[:noise_frames]))

        # Apply spectral gate
        threshold = noise_profile * 1.5
        audio_gated = audio.copy()
        audio_gated[np.abs(audio) < threshold] *= 0.5

        return audio_gated

    def normalize_loudness(self, audio: np.ndarray, target_loudness: float = -14.0) -> np.ndarray:
        """Normalize audio loudness to target (LUFS)."""
        try:
            import pyloudnorm

            meter = pyloudnorm.Meter(self.sample_rate)
            loudness = meter.integrated_loudness(audio)

            if not np.isfinite(loudness):
                return audio

            normalized = pyloudnorm.normalize.loudness(audio, loudness, target_loudness)
            return normalized

        except ImportError:
            # Fallback: simple peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio * (0.9 / peak)
            return audio
        except Exception as e:
            logger.debug("Loudness normalization failed: {}", e)
            return audio

    def mix_audio(
        self,
        dubbed_audio: np.ndarray,
        original_audio: np.ndarray,
        dubbed_weight: float = 1.0,
        sfx_weight: float = 0.3,
    ) -> np.ndarray:
        """Mix dubbed speech with separated background SFX/music."""
        if len(dubbed_audio) != len(original_audio):
            min_len = min(len(dubbed_audio), len(original_audio))
            dubbed_audio = dubbed_audio[:min_len]
            original_audio = original_audio[:min_len]

        mixed = (dubbed_audio * dubbed_weight) + (original_audio * sfx_weight)

        # Prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val

        return mixed

    def add_reverb(self, audio: np.ndarray, room_scale: float = 0.5) -> np.ndarray:
        """Add subtle reverb for naturalness using comb filter."""
        try:
            delay_samples = int(0.05 * self.sample_rate)  # 50ms
            decay = room_scale

            filtered = audio.copy()
            for i in range(delay_samples, len(filtered)):
                filtered[i] += decay * filtered[i - delay_samples]

            return filtered / (1 + decay)

        except Exception as e:
            logger.debug("Reverb failed: {}", e)
            return audio
