#!/usr/bin/env python3
"""
End-to-End Dubbing Orchestrator — Phase 3

Complete pipeline:
  1. Extract audio from video
  2. Transcribe (ASR) → segment-level with timestamps
  3. Translate each segment (SeamlessM4T)
  4. Synthesize speech per segment (TTS) with voice cloning
  5. Time-stretch each segment to match original timing
  6. Stitch segments + post-process audio
  7. Synchronize lip movements (MuseTalk)
  8. Mix audio + encode output video

Usage:
    from src.pipeline import DubbingPipeline

    pipeline = DubbingPipeline()
    result = pipeline.dub_video(
        video_path="input.mp4",
        reference_speaker_audio="speaker.wav",
        emotion="neutral",
        output_path="dubbed.mp4"
    )
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from src.inference import (
    ASRInference,
    TranslationInference,
    TTSInference,
    LipSyncInference,
    AudioPostProcessor,
)
from src.utils.config import load_config
from src.utils.helpers import (
    extract_audio_from_video,
    get_video_info,
    time_stretch_audio,
    load_audio,
    save_audio,
    timer,
    free_gpu_memory,
    log_voice_consent,
)

# Dialect → SeamlessM4T language code mapping
DIALECT_MAP = {
    "eastern": "hye",   # Eastern Armenian (ISO 639-3)
    "western": "hyw",   # Western Armenian (ISO 639-3)
    "hye": "hye",
    "hyw": "hyw",
}


class DubbingPipeline:
    """Complete video dubbing pipeline with segment-level alignment."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_override_path: Optional[str] = None,
    ):
        """Initialize pipeline with configuration."""
        self.config = load_config(config_path=config_path, override_path=config_override_path)

        project_cfg = self.config.get("project", {})
        audio_cfg = self.config.get("audio", {})
        asr_cfg = self.config.get("asr", {})
        whisper_cfg = asr_cfg.get("whisper", {})
        translation_cfg = self.config.get("translation", {})
        tts_cfg = self.config.get("tts", {})
        lipsync_cfg = self.config.get("lipsync", {})
        inference_cfg = self.config.get("inference", {})

        requested_device = project_cfg.get("device", "cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = requested_device

        logger.info("Using device: {}", self.device)

        # Read quantization setting from config
        inference_cfg = self.config.get("inference", {})
        quant_bits = 0
        if inference_cfg.get("enable_quantization"):
            quant_bits = inference_cfg.get("quantization_bits", 0)

        self.unload_models_after_stage = inference_cfg.get("unload_models_after_stage", False)
        self.max_input_video_sec = inference_cfg.get("max_input_video_sec")
        self.lipsync_enabled = lipsync_cfg.get("enabled", True)
        self.background_separation_enabled = audio_cfg.get("demucs", {}).get("enabled", True)
        self.loudness_target = audio_cfg.get("loudness_target", -14.0)
        self.timing_method = self.config.get("timing", {}).get("method", "rubberband")
        self.max_stretch_ratio = self.config.get("timing", {}).get("max_stretch_ratio", 1.25)
        self.min_compress_ratio = self.config.get("timing", {}).get("min_compress_ratio", 0.80)

        whisper_model = whisper_cfg.get("model", "large-v3")
        if "/" in whisper_model:
            whisper_model_id = whisper_model
        else:
            whisper_model_id = f"openai/whisper-{whisper_model}"

        translation_model_source = self._resolve_model_source(
            translation_cfg.get("model_path"),
            translation_cfg.get("model", "facebook/seamless-m4t-v2-large"),
        )

        # Initialize inference modules (lazy-loaded on first use)
        self.asr = ASRInference(
            model_path=Path(whisper_cfg.get("model_path", "models/asr/whisper-hy-full")),
            model_id=whisper_model_id,
            device=self.device,
            use_fp16=project_cfg.get("dtype", "float16") != "float32",
            quantize_bits=quant_bits,
            language=whisper_cfg.get("language", "hy"),
            task=whisper_cfg.get("task", "transcribe"),
            chunk_length_s=whisper_cfg.get("chunk_length_s", 30),
            batch_size=whisper_cfg.get("batch_size", 8),
        )
        self.translator = TranslationInference(
            model_id=translation_model_source,
            device=self.device,
            dtype=translation_cfg.get("dtype", project_cfg.get("dtype", "float16")),
            num_beams=translation_cfg.get("num_beams", 5),
            max_new_tokens=translation_cfg.get("max_length", 512),
        )
        self.tts = TTSInference(
            model_path=Path(tts_cfg.get("fish_speech", {}).get("model_path", "models/tts/fish-speech-hy")),
            device=self.device,
            sample_rate=audio_cfg.get("sample_rate", 44100),
            preferred_backend=tts_cfg.get("backend", "auto"),
        )
        self.lip_sync = LipSyncInference(
            model_path=Path(lipsync_cfg.get("model_path", "models/lipsync/MuseTalk")),
            device=self.device,
        )
        self.audio_processor = AudioPostProcessor(
            sample_rate=audio_cfg.get("sample_rate", 44100),
            enable_source_separation=self.background_separation_enabled,
        )

        # Ethics config
        self.ethics = self.config.get("ethics", {})

        self.sr = audio_cfg.get("sample_rate", 44100)
        self.temp_dir = Path(self.config.get("paths", {}).get("temp_dir", "outputs/temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_model_source(local_path: Optional[str], fallback_model_id: str) -> str:
        """Prefer a local model path if it exists, otherwise use the remote model id."""
        if local_path:
            model_path = Path(local_path)
            if model_path.exists():
                return str(model_path)
        return fallback_model_id

    def _maybe_unload(self, module, label: str) -> None:
        """Unload a model between stages when using memory-constrained profiles."""
        if not self.unload_models_after_stage:
            return

        try:
            module.free_memory()
            free_gpu_memory()
            logger.info("Freed model memory after {} stage", label)
        except Exception as e:
            logger.debug("Memory cleanup after {} skipped: {}", label, e)

    def dub_video(
        self,
        video_path: str,
        reference_speaker_audio: Optional[str] = None,
        emotion: str = "neutral",
        output_path: str = "dubbed_output.mp4",
        keep_background: bool = True,
        skip_lipsync: bool = False,
        src_lang: str = "eng",
        tgt_lang: str = "hye",
        dialect: str = "eastern",
    ) -> dict:
        """Run the complete dubbing pipeline.

        Args:
            video_path: Input video file path.
            reference_speaker_audio: Optional reference audio for voice cloning.
            emotion: Emotion style (neutral, happy, sad, angry, excited, calm).
            output_path: Where to save the dubbed video.
            keep_background: Whether to keep original background audio/SFX.
            skip_lipsync: Skip lip-sync step (faster).
            src_lang: Source language code.
            tgt_lang: Target language code.
            dialect: Armenian dialect ("eastern" or "western").

        Returns:
            Dict with status, output_video, transcription, duration_sec.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            logger.error("Video file not found: {}", video_path)
            return {"error": "File not found"}

        if self.max_input_video_sec:
            try:
                video_info = get_video_info(video_path)
                video_duration = video_info.get("duration", 0.0)
                if video_duration > self.max_input_video_sec:
                    return {
                        "error": (
                            f"Input video is {video_duration:.1f}s, which exceeds the "
                            f"configured Colab/runtime limit of {self.max_input_video_sec}s"
                        )
                    }
            except Exception as e:
                logger.warning("Could not inspect input video duration: {}", e)

        # Resolve dialect to language code
        tgt_lang = DIALECT_MAP.get(dialect, tgt_lang)

        # Log voice consent if using voice cloning
        if reference_speaker_audio and self.ethics.get("consent_required", False):
            speaker_id = Path(reference_speaker_audio).stem
            log_voice_consent(
                speaker_id=speaker_id,
                consent_given=True,
                consent_log=self.ethics.get("consent_log_path", "logs/voice_consent.json"),
            )

        logger.info("=" * 60)
        logger.info("Starting dubbing: {}", video_path.name)
        logger.info("  Language: {} → {} (dialect: {})", src_lang, tgt_lang, dialect)
        logger.info("  Emotion: {}", emotion)
        logger.info("  Lip-sync: {}", "enabled" if not skip_lipsync else "disabled")
        logger.info("=" * 60)

        with timer("Complete dubbing"):
            try:
                # Step 1: Extract audio from video
                logger.info("[Step 1/8] Extracting audio from video...")
                audio_path = self._extract_audio(video_path)

                # Step 2: Transcribe (ASR) with timestamps
                logger.info("[Step 2/8] Transcribing audio (ASR)...")
                transcription = self._transcribe_audio(audio_path, src_lang=src_lang)
                if "error" in transcription:
                    return {"error": f"ASR failed: {transcription['error']}"}
                self._maybe_unload(self.asr, "ASR")

                segments = transcription.get("segments", [])
                full_text = transcription.get("text", "")
                logger.info("  Transcribed {} segments, {} chars", len(segments), len(full_text))

                if not full_text.strip():
                    logger.warning("ASR produced empty transcription — nothing to dub")
                    return {
                        "status": "success",
                        "output_video": str(output_path),
                        "transcription": "",
                        "translated_text": "",
                        "n_segments": 0,
                        "emotion": emotion,
                        "duration_sec": 0.0,
                        "warning": "No speech detected in audio",
                    }

                # Step 3: Translate each segment
                logger.info("[Step 3/8] Translating {} segments...", len(segments))
                translated_segments = self._translate_segments(segments, src_lang, tgt_lang)
                self._maybe_unload(self.translator, "translation")

                # Step 4: Synthesize speech per segment (TTS)
                logger.info("[Step 4/8] Synthesizing speech (TTS)...")
                segment_audios = self._synthesize_segments(
                    translated_segments,
                    reference_speaker_audio=reference_speaker_audio,
                    emotion=emotion,
                    language=tgt_lang,
                )
                self._maybe_unload(self.tts, "TTS")

                # Step 5: Time-stretch segments and stitch into single audio
                logger.info("[Step 5/8] Aligning segment durations...")
                original_audio, _ = load_audio(audio_path, sr=self.sr)
                original_duration = len(original_audio) / self.sr
                dubbed_audio = self._align_and_stitch_segments(
                    segment_audios, translated_segments, original_duration
                )

                # Step 6: Post-process audio (denoise, normalize, mix)
                logger.info("[Step 6/8] Post-processing audio...")
                final_audio = self._process_audio(
                    dubbed_audio,
                    original_audio_path=audio_path if keep_background else None,
                )

                # Step 7: Lip-sync (optional)
                if not skip_lipsync:
                    logger.info("[Step 7/8] Synchronizing lip movements...")
                    fused_video = self._apply_lipsync(video_path, final_audio)
                    self._maybe_unload(self.lip_sync, "lip-sync")
                else:
                    logger.info("[Step 7/8] Skipping lip-sync")
                    fused_video = video_path

                # Step 8: Mix audio + encode final video
                logger.info("[Step 8/8] Encoding final video...")
                output_video = self._mix_and_encode(
                    str(fused_video), final_audio, output_path,
                )

                logger.info("Dubbing complete: {}", output_path)

                return {
                    "status": "success",
                    "output_video": str(output_video),
                    "transcription": full_text,
                    "translated_text": " ".join(s["text"] for s in translated_segments),
                    "n_segments": len(segments),
                    "emotion": emotion,
                    "duration_sec": original_duration,
                }

            except Exception as e:
                logger.error("Pipeline failed: {}", e)
                import traceback
                logger.error(traceback.format_exc())
                return {"error": str(e)}

    # ========================================================================
    # Pipeline Steps
    # ========================================================================

    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio track from video."""
        output_audio = self.temp_dir / f"{video_path.stem}_extracted.wav"

        if output_audio.exists():
            logger.info("  Using cached audio: {}", output_audio.name)
            return output_audio

        return extract_audio_from_video(video_path, output_audio, sr=self.sr)

    def _transcribe_audio(self, audio_path: Path, src_lang: str = "eng") -> dict:
        """Transcribe audio with segment-level timestamps.

        Uses the source language to guide Whisper (e.g. 'eng' for English input).
        """
        self.asr.load()
        return self.asr.transcribe(str(audio_path), language=src_lang)

    def _translate_segments(
        self, segments: list, src_lang: str, tgt_lang: str
    ) -> list:
        """Translate each ASR segment. If src==tgt, pass through."""
        if src_lang == tgt_lang:
            logger.info("  Same language — skipping translation")
            return segments

        self.translator.load()
        translated = self.translator.translate_segments(segments, src_lang, tgt_lang)

        for i, seg in enumerate(translated):
            if i < 3:  # Log first few for debugging
                logger.debug("  [{}] '{}' → '{}'", i, seg.get("src_text", "")[:40], seg["text"][:40])

        return translated

    def _synthesize_segments(
        self,
        segments: list,
        reference_speaker_audio: Optional[str] = None,
        emotion: str = "neutral",
        language: str = "hye",
    ) -> list:
        """Synthesize speech for each translated segment.

        Returns list of dicts with "audio" (np.ndarray) and "sample_rate".
        """
        self.tts.load()

        results = []
        for i, seg in enumerate(segments):
            text = seg["text"]
            if not text.strip():
                # Silent gap
                gap_duration = max(0.1, seg.get("end", 0) - seg.get("start", 0))
                results.append({
                    "audio": np.zeros(int(gap_duration * self.sr), dtype=np.float32),
                    "sample_rate": self.sr,
                    "duration_sec": gap_duration,
                })
                continue

            tts_result = self.tts.synthesize(
                text=text,
                reference_audio_path=reference_speaker_audio,
                emotion=emotion,
                language=language[:2],  # "hye" → "hy"
            )
            results.append(tts_result)

            if i % 10 == 0 and i > 0:
                logger.info("  Synthesized {}/{} segments", i, len(segments))

        logger.info("  Synthesized {} segments total", len(results))
        return results

    def _align_and_stitch_segments(
        self,
        segment_audios: list,
        segments: list,
        total_duration: float,
    ) -> np.ndarray:
        """Align synthesized segments to original timestamps and stitch together.

        Each segment is time-stretched to fit the original segment's time slot,
        then placed at the correct offset in the output buffer.
        """
        total_samples = int(total_duration * self.sr)
        output = np.zeros(total_samples, dtype=np.float32)

        for i, (seg_audio, seg_info) in enumerate(zip(segment_audios, segments)):
            audio = seg_audio.get("audio", np.array([], dtype=np.float32))
            if len(audio) == 0:
                continue

            # Resample if needed
            audio_sr = seg_audio.get("sample_rate", self.sr)
            if audio_sr != self.sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=self.sr)

            start_sec = seg_info.get("start", 0.0)
            end_sec = seg_info.get("end", 0.0)
            target_duration = end_sec - start_sec

            if target_duration > 0.05 and len(audio) > 100:
                # Time-stretch this segment to fit the time slot
                audio_duration = len(audio) / self.sr
                ratio = target_duration / audio_duration

                if abs(ratio - 1.0) > 0.05:
                    ratio = max(self.min_compress_ratio, min(self.max_stretch_ratio, ratio))
                    # Use rubberband for quality stretching
                    try:
                        tmp_in = self.temp_dir / f"seg_{i}_in.wav"
                        tmp_out = self.temp_dir / f"seg_{i}_out.wav"
                        save_audio(audio, tmp_in, sr=self.sr)
                        time_stretch_audio(
                            tmp_in,
                            tmp_out,
                            target_duration=target_duration,
                            method=self.timing_method,
                        )
                        audio, _ = load_audio(tmp_out, sr=self.sr)
                    except Exception:
                        # Fallback: simple resampling
                        target_samples = int(target_duration * self.sr)
                        if target_samples > 0:
                            indices = np.linspace(0, len(audio) - 1, target_samples).astype(int)
                            audio = audio[indices]

            # Place in output buffer
            start_sample = int(start_sec * self.sr)
            end_sample = start_sample + len(audio)

            if end_sample > total_samples:
                audio = audio[:total_samples - start_sample]
                end_sample = total_samples

            if start_sample < total_samples and len(audio) > 0:
                output[start_sample:start_sample + len(audio)] = audio

        return output

    def _process_audio(
        self,
        dubbed_audio: np.ndarray,
        original_audio_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Post-process: denoise, normalize, optionally mix with background."""
        # Denoise
        audio = self.audio_processor.denoise_audio(dubbed_audio)

        # Normalize loudness
        audio = self.audio_processor.normalize_loudness(audio, target_loudness=self.loudness_target)

        # Mix with background if requested
        if original_audio_path:
            if not self.background_separation_enabled:
                logger.info("Background preservation disabled in config; skipping source separation")
                return audio

            try:
                bg_audio, _ = load_audio(original_audio_path, sr=self.sr)

                # Separate sources to get just the accompaniment
                separated = self.audio_processor.separate_sources(bg_audio)
                accompaniment = separated.get("accompaniment", bg_audio)

                audio = self.audio_processor.mix_audio(
                    audio,
                    accompaniment,
                    dubbed_weight=1.0,
                    sfx_weight=0.2,
                )
            except Exception as e:
                logger.warning("Background mixing failed: {}", e)

        return audio

    def _apply_lipsync(self, video_path: Path, audio: np.ndarray) -> Path:
        """Apply lip-sync using MuseTalk."""
        if not self.lipsync_enabled:
            logger.info("Lip-sync disabled in config profile")
            return video_path

        temp_audio = self.temp_dir / f"{video_path.stem}_dubbed_audio.wav"
        save_audio(audio, temp_audio, sr=self.sr)

        output_video = self.temp_dir / f"{video_path.stem}_lipsync.mp4"

        self.lip_sync.load()
        result = self.lip_sync.inpaint_mouth(
            str(video_path),
            str(temp_audio),
            output_path=str(output_video),
        )

        output = Path(result.get("output", str(video_path)))
        if output.exists() and result.get("status") == "success":
            return output

        logger.warning("Lip-sync skipped — using original video")
        return video_path

    def _mix_and_encode(
        self,
        video_path: str,
        audio: np.ndarray,
        output_path: Path,
    ) -> Path:
        """Mix dubbed audio into video, apply watermark if configured, and encode."""
        import subprocess

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save audio
        temp_audio = self.temp_dir / "final_audio.wav"
        save_audio(audio, temp_audio, sr=self.sr)

        # Build video filter for watermark
        vf_filters = []
        if self.ethics.get("add_watermark", False):
            wm_text = self.ethics.get("watermark_text", "AI-Dubbed")
            wm_opacity = self.ethics.get("watermark_opacity", 0.3)
            # FFmpeg drawtext filter — bottom-right corner, semi-transparent
            vf_filters.append(
                f"drawtext=text='{wm_text}':fontsize=18:"
                f"fontcolor=white@{wm_opacity}:"
                f"x=w-tw-10:y=h-th-10"
            )

        # FFmpeg: replace audio track (+ optional watermark)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", str(temp_audio),
        ]

        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])
            cmd.extend(["-c:v", "libx264"])
        else:
            cmd.extend(["-c:v", "libx264"])

        cmd.extend([
            "-crf", str(self.config.get("video", {}).get("output_crf", 18)),
            "-preset", self.config.get("video", {}).get("output_preset", "medium"),
            "-c:a", "aac",
            "-b:a", self.config.get("video", {}).get("output_audio_bitrate", "192k"),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
            logger.info("Video encoded: {}", output_path)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg failed: {}", e.stderr[:500] if e.stderr else str(e))
            return Path(video_path)
        except Exception as e:
            logger.error("Encoding failed: {}", e)
            return Path(video_path)

    def cleanup_temp(self):
        """Remove temporary files."""
        import shutil
        if self.temp_dir.exists():
            for f in self.temp_dir.iterdir():
                try:
                    if f.is_file():
                        f.unlink()
                except OSError:
                    pass


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Armenian Video Dubbing Pipeline")
    parser.add_argument("video", type=str, help="Input video file")
    parser.add_argument("--reference-speaker", type=str, default=None,
                        help="Reference speaker audio for voice cloning")
    parser.add_argument("--emotion", type=str, default="neutral",
                        choices=["neutral", "happy", "sad", "angry", "excited", "calm"])
    parser.add_argument("--output", type=str, default="dubbed_output.mp4",
                        help="Output video path")
    parser.add_argument("--skip-lipsync", action="store_true",
                        help="Skip lip-sync step (faster)")
    parser.add_argument("--no-background", action="store_true",
                        help="Don't mix background SFX/music")
    parser.add_argument("--src-lang", type=str, default="eng",
                        help="Source language (default: eng)")
    parser.add_argument("--tgt-lang", type=str, default="hye",
                        help="Target language (default: hye = Eastern Armenian)")
    parser.add_argument("--dialect", type=str, default="eastern",
                        choices=["eastern", "western"],
                        help="Armenian dialect (default: eastern)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config file path")
    parser.add_argument("--config-override", type=str, default=None,
                        help="Optional override config merged on top of the base config")

    args = parser.parse_args()

    from src.utils.logger import setup_logger
    setup_logger()

    pipeline = DubbingPipeline(
        config_path=args.config,
        config_override_path=args.config_override,
    )
    result = pipeline.dub_video(
        video_path=args.video,
        reference_speaker_audio=args.reference_speaker,
        emotion=args.emotion,
        output_path=args.output,
        keep_background=not args.no_background,
        skip_lipsync=args.skip_lipsync,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        dialect=args.dialect,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
