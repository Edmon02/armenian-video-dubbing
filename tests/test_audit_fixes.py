#!/usr/bin/env python3
"""Regression tests for fixes from docs/research/2026-10-dubbing-audit-and-plan.md."""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestShortLinesKept:
    @pytest.mark.parametrize("text", ["No!", "Yes.", "Why?", "Okay"])
    def test_short_dialogue_is_not_hallucination(self, text):
        from src.inference import ASRInference
        assert ASRInference._is_hallucinated(text) is False

    def test_empty_and_repetitive_are_hallucination(self):
        from src.inference import ASRInference
        assert ASRInference._is_hallucinated("") is True
        assert ASRInference._is_hallucinated("   ") is True
        assert ASRInference._is_hallucinated(" ".join(["you"] * 20)) is True


class TestTranslationFailureIsSilent:
    def _failing_translator(self):
        from src.inference import TranslationInference
        t = TranslationInference(device="cpu")
        t.model = MagicMock()
        t.processor = MagicMock(side_effect=RuntimeError("boom"))
        return t

    def test_failure_never_returns_source_text(self):
        result = self._failing_translator().translate("Hello there", "eng", "hye")
        assert result["tgt_text"] == ""
        assert "error" in result

    def test_failed_segment_is_flagged(self):
        segs = [{"text": "Hello there", "start": 0.0, "end": 1.0}]
        out = self._failing_translator().translate_segments(segs, "eng", "hye")
        assert out[0]["text"] == ""
        assert "translation_error" in out[0]

    def test_pipeline_reports_failed_segments(self):
        from src.pipeline.pipeline import DubbingPipeline
        segs = [{"text": "", "translation_error": "boom"}, {"text": "Բարեւ"}]
        assert "1/2" in DubbingPipeline._translation_warning(segs)
        assert DubbingPipeline._translation_warning([{"text": "Բարեւ"}]) is None


class TestStretchClamp:
    def test_short_tts_is_not_stretched_past_max_ratio(self):
        from src.pipeline import DubbingPipeline
        pipeline = DubbingPipeline()
        sr = pipeline.sr
        segments = [{"text": "a", "start": 0.0, "end": 3.0}]
        audios = [{"audio": np.full(sr, 0.5, dtype=np.float32), "sample_rate": sr}]

        out = pipeline._align_and_stitch_segments(audios, segments, total_duration=3.0)

        voiced_sec = np.count_nonzero(np.abs(out) > 1e-3) / sr
        expected = 1.0 * pipeline.max_stretch_ratio
        assert voiced_sec == pytest.approx(expected, rel=0.08)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestFfmpegStretchDirection:
    def test_atempo_lengthens_when_target_is_longer(self):
        from src.utils.helpers import get_audio_duration, save_audio, time_stretch_audio
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "in.wav", Path(d) / "out.wav"
            t = np.arange(16000) / 16000
            save_audio((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), src, sr=16000)
            time_stretch_audio(src, dst, target_duration=1.5, method="ffmpeg")
            assert get_audio_duration(dst) == pytest.approx(1.5, rel=0.05)


class TestSmallHelpers:
    @pytest.mark.parametrize("rate,expected", [
        ("25/1", 25.0), ("30000/1001", 29.97), ("0/0", 25.0), ("__import__('os')", 25.0),
    ])
    def test_parse_frame_rate(self, rate, expected):
        from src.utils.helpers import _parse_frame_rate
        assert _parse_frame_rate(rate) == pytest.approx(expected, rel=1e-3)

    def test_filter_value_escaping(self):
        from src.pipeline.pipeline import _escape_filter_value
        escaped = _escape_filter_value("/tmp/a:b,c[d]'e")
        for raw in (":", ",", "[", "]", "'"):
            assert f"\\{raw}" in escaped


class TestNoFakeMetrics:
    @pytest.fixture(autouse=True)
    def _metrics_package_deps(self):
        for module in ("jiwer", "psutil"):
            pytest.importorskip(module)

    def _translation_computer(self):
        from scripts.evaluation.metrics.translation_metrics import TranslationQualityComputer
        computer = TranslationQualityComputer.__new__(TranslationQualityComputer)
        computer.device = "cpu"
        computer.comet_model = MagicMock()
        computer.comet_model.predict.return_value = MagicMock(scores=[0.61, 0.73])
        return computer

    def test_comet_uses_model_scores(self):
        result = self._translation_computer().compute_comet_batch(
            ["a", "b"], ["x", "y"], ["rx", "ry"]
        )
        assert result["scores"] == [0.61, 0.73]
        assert result["comet_score"] == pytest.approx(0.67)

    def test_comet_without_reference_is_an_error(self):
        computer = self._translation_computer()
        assert "error" in computer.compute_comet_score("a", "x")
        computer.comet_model = None
        assert "error" in computer.compute_comet_batch(["a"], ["x"], ["r"])

    def test_lipsync_stub_returns_error_not_numbers(self):
        from scripts.evaluation.metrics.lipsync_metrics import LipSyncMetricsComputer
        computer = LipSyncMetricsComputer(device="cpu")
        c = computer.compute_lse_c_metric("v.mp4", "a.wav")
        d = computer.compute_lse_d_metric("v.mp4", "a.wav")
        assert "lse_c" not in c and "error" in c
        assert "lse_d" not in d and "error" in d
