#!/usr/bin/env python3
"""
Integration tests for the Armenian Video Dubbing pipeline.

Tests validate that each inference module can be instantiated and that
the pipeline orchestration logic works correctly (with mocked models
when GPU/models are not available).
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Unit Tests — Inference Module Instantiation
# ============================================================================

class TestASRInference:
    def test_init(self):
        from src.inference import ASRInference
        asr = ASRInference(device="cpu")
        assert asr.model is None
        assert asr.device == "cpu"

    def test_transcribe_returns_expected_keys(self):
        """If model loads (needs transformers), verify output structure."""
        from src.inference import ASRInference
        asr = ASRInference(device="cpu")

        # Mock the pipeline so we don't download the model
        asr.model = MagicMock()
        asr._pipe = MagicMock(return_value={
            "text": "Բարև ձեզ",
            "chunks": [
                {"text": "Բարև", "timestamp": (0.0, 1.0)},
                {"text": "ձեզ", "timestamp": (1.0, 2.0)},
            ],
        })

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf
            audio = np.random.randn(16000).astype(np.float32) * 0.1
            sf.write(f.name, audio, 16000)

            with patch("librosa.get_duration", return_value=1.0):
                result = asr.transcribe(f.name)

            os.unlink(f.name)

        assert "text" in result
        assert "segments" in result
        assert isinstance(result["segments"], list)


class TestTranslationInference:
    def test_init(self):
        from src.inference import TranslationInference
        t = TranslationInference(device="cpu")
        assert t.model is None

    def test_empty_text(self):
        from src.inference import TranslationInference
        t = TranslationInference(device="cpu")
        result = t.translate("", src_lang="eng", tgt_lang="hye")
        assert result["tgt_text"] == ""

    def test_translate_segments(self):
        from src.inference import TranslationInference
        t = TranslationInference(device="cpu")

        # Mock the model to return a known translation
        t.model = MagicMock()
        t.processor = MagicMock()
        t.processor.return_value = {"input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))}
        t.processor.batch_decode = MagicMock(return_value=["Բարև աշխարհ"])
        t.model.generate = MagicMock(return_value=MagicMock())

        segments = [
            {"text": "Hello world", "start": 0.0, "end": 2.0},
            {"text": "How are you", "start": 2.0, "end": 4.0},
        ]

        result = t.translate_segments(segments, "eng", "hye")
        assert len(result) == 2
        assert "text" in result[0]
        assert "start" in result[0]
        assert "end" in result[0]


class TestTTSInference:
    def test_init(self):
        from src.inference import TTSInference
        tts = TTSInference(device="cpu")
        assert tts.backend is None

    def test_empty_text(self):
        from src.inference import TTSInference
        tts = TTSInference(device="cpu")
        tts.backend = "edge-tts"  # Pretend loaded
        result = tts.synthesize("", emotion="neutral")
        assert "audio" in result
        assert isinstance(result["audio"], np.ndarray)

    def test_speaker_embedding_fallback(self):
        from src.inference import TTSInference
        tts = TTSInference(device="cpu")
        audio = np.random.randn(16000).astype(np.float32)
        emb = tts.extract_speaker_embedding(audio)
        assert isinstance(emb, np.ndarray)


class TestLipSyncInference:
    def test_init(self):
        from src.inference import LipSyncInference
        ls = LipSyncInference(device="cpu")
        assert ls.available is False

    def test_graceful_skip_when_not_installed(self):
        from src.inference import LipSyncInference
        ls = LipSyncInference(device="cpu")
        result = ls.inpaint_mouth("test.mp4", "test.wav")
        assert result["status"] == "skipped"


class TestAudioPostProcessor:
    def test_denoise(self):
        from src.inference import AudioPostProcessor
        proc = AudioPostProcessor(sample_rate=16000)
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        result = proc.denoise_audio(audio)
        assert len(result) == len(audio)

    def test_mix_audio(self):
        from src.inference import AudioPostProcessor
        proc = AudioPostProcessor(sample_rate=16000)
        a = np.random.randn(16000).astype(np.float32) * 0.5
        b = np.random.randn(16000).astype(np.float32) * 0.5
        mixed = proc.mix_audio(a, b, dubbed_weight=1.0, sfx_weight=0.3)
        assert len(mixed) == len(a)
        assert np.max(np.abs(mixed)) <= 1.0

    def test_mix_different_lengths(self):
        from src.inference import AudioPostProcessor
        proc = AudioPostProcessor(sample_rate=16000)
        a = np.random.randn(16000).astype(np.float32) * 0.5
        b = np.random.randn(8000).astype(np.float32) * 0.5
        mixed = proc.mix_audio(a, b)
        assert len(mixed) == 8000

    def test_normalize_loudness_fallback(self):
        from src.inference import AudioPostProcessor
        proc = AudioPostProcessor(sample_rate=16000)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = proc.normalize_loudness(audio)
        assert len(result) == len(audio)

    def test_reverb(self):
        from src.inference import AudioPostProcessor
        proc = AudioPostProcessor(sample_rate=16000)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = proc.add_reverb(audio, room_scale=0.3)
        assert len(result) == len(audio)


# ============================================================================
# Integration Test — Pipeline
# ============================================================================

class TestDubbingPipeline:
    def test_init(self):
        from src.pipeline import DubbingPipeline
        pipeline = DubbingPipeline()
        assert pipeline.device in ("cuda", "cpu")
        assert pipeline.sr > 0

    def test_missing_video_file(self):
        from src.pipeline import DubbingPipeline
        pipeline = DubbingPipeline()
        result = pipeline.dub_video("nonexistent.mp4")
        assert "error" in result

    def test_align_and_stitch(self):
        """Test segment alignment without models."""
        from src.pipeline import DubbingPipeline
        pipeline = DubbingPipeline()

        segments = [
            {"text": "hello", "start": 0.0, "end": 1.0},
            {"text": "world", "start": 1.5, "end": 2.5},
        ]
        seg_audios = [
            {"audio": np.random.randn(int(pipeline.sr * 0.8)).astype(np.float32) * 0.1, "sample_rate": pipeline.sr},
            {"audio": np.random.randn(int(pipeline.sr * 0.9)).astype(np.float32) * 0.1, "sample_rate": pipeline.sr},
        ]

        result = pipeline._align_and_stitch_segments(seg_audios, segments, total_duration=3.0)
        assert len(result) == int(3.0 * pipeline.sr)

    def test_config_override_applies(self):
        """Override config should affect runtime-friendly settings."""
        from src.pipeline import DubbingPipeline

        override = """
inference:
  max_input_video_sec: 12
audio:
  demucs:
    enabled: false
tts:
  backend: edge-tts
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(override)
            override_path = f.name

        try:
            pipeline = DubbingPipeline(config_override_path=override_path)
            assert pipeline.max_input_video_sec == 12
            assert pipeline.audio_processor.enable_source_separation is False
            assert pipeline.tts.preferred_backend == "edge-tts"
        finally:
            os.unlink(override_path)


# ============================================================================
# Utility Tests
# ============================================================================

class TestHelpers:
    def test_load_save_audio(self):
        from src.utils.helpers import load_audio, save_audio
        audio = np.random.randn(16000).astype(np.float32) * 0.5

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio(audio, f.name, sr=16000)
            loaded, sr = load_audio(f.name, sr=16000)
            os.unlink(f.name)

        assert sr == 16000
        assert len(loaded) == len(audio)
        np.testing.assert_allclose(loaded, audio, atol=1e-4)

    def test_file_hash(self):
        from src.utils.helpers import file_hash
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            h = file_hash(f.name)
            os.unlink(f.name)

        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256

    def test_timer(self):
        from src.utils.helpers import timer
        with timer("test"):
            pass  # Should not raise


class TestConfig:
    def test_config_loads(self):
        import yaml
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)

        assert "project" in config
        assert "asr" in config
        assert "translation" in config
        assert "tts" in config
        assert "lipsync" in config
        assert "audio" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
