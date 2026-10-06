#!/usr/bin/env python3
"""
Common helper utilities for Armenian Video Dubbing AI.

Audio I/O, timing, GPU management, file operations, etc.
"""

import gc
import hashlib
import json
import subprocess
import tempfile
import time
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf
import torch
from loguru import logger


# ============================================================================
# Audio I/O
# ============================================================================

def load_audio(path: str | Path, sr: int = 44100, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio file, resampling and converting to mono if needed.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    import librosa

    path = str(path)
    audio, orig_sr = librosa.load(path, sr=sr, mono=mono)
    return audio, sr


def save_audio(audio: np.ndarray, path: str | Path, sr: int = 44100) -> Path:
    """Save audio array to WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    logger.debug("Saved audio: {} ({:.1f}s, {}Hz)", path.name, len(audio) / sr, sr)
    return path


def get_audio_duration(path: str | Path) -> float:
    """Get audio file duration in seconds."""
    info = sf.info(str(path))
    return info.duration


# ============================================================================
# Video helpers
# ============================================================================

def _parse_frame_rate(rate: str) -> float:
    """Parse an ffprobe rate such as "30000/1001" without eval()."""
    try:
        value = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 25.0
    return value if value > 0 else 25.0


def get_video_info(path: str | Path) -> dict:
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    video_stream = next(
        (s for s in info.get("streams", []) if s["codec_type"] == "video"), None
    )
    audio_stream = next(
        (s for s in info.get("streams", []) if s["codec_type"] == "audio"), None
    )

    return {
        "duration": float(info["format"].get("duration", 0)),
        "width": int(video_stream["width"]) if video_stream else 0,
        "height": int(video_stream["height"]) if video_stream else 0,
        "fps": _parse_frame_rate(video_stream.get("r_frame_rate", "25/1")) if video_stream else 25,
        "video_codec": video_stream.get("codec_name") if video_stream else None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "audio_sr": int(audio_stream.get("sample_rate", 44100)) if audio_stream else 44100,
    }


def extract_audio_from_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    sr: int = 44100,
) -> Path:
    """Extract audio track from video file."""
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    logger.debug("Extracted audio: {} -> {}", video_path.name, output_path.name)
    return output_path


# ============================================================================
# GPU / Memory management
# ============================================================================

def get_gpu_memory_info() -> dict:
    """Get GPU memory info in GB."""
    if not torch.cuda.is_available():
        return {"total": 0, "used": 0, "free": 0}

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    used = torch.cuda.memory_allocated(0) / 1e9
    cached = torch.cuda.memory_reserved(0) / 1e9
    return {"total": total, "used": used, "cached": cached, "free": total - cached}


def free_gpu_memory() -> None:
    """Force free GPU memory."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU memory freed. Current: {:.1f}GB used", get_gpu_memory_info()["used"])


@contextmanager
def gpu_memory_guard(label: str = "operation") -> Generator[None, None, None]:
    """Context manager that logs GPU memory before/after and cleans up."""
    if torch.cuda.is_available():
        before = get_gpu_memory_info()
        logger.debug("[{}] GPU memory before: {:.1f}GB used", label, before["used"])

    try:
        yield
    finally:
        if torch.cuda.is_available():
            free_gpu_memory()
            after = get_gpu_memory_info()
            logger.debug("[{}] GPU memory after cleanup: {:.1f}GB used", label, after["used"])


# ============================================================================
# Timing
# ============================================================================

@contextmanager
def timer(label: str = "operation") -> Generator[None, None, None]:
    """Context manager that logs elapsed time."""
    start = time.perf_counter()
    logger.info("[{}] Starting...", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if elapsed > 60:
            logger.info("[{}] Completed in {:.1f} min", label, elapsed / 60)
        else:
            logger.info("[{}] Completed in {:.1f}s", label, elapsed)


# ============================================================================
# File helpers
# ============================================================================

def file_hash(path: str | Path, algo: str = "sha256") -> str:
    """Compute file hash."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def temp_path(suffix: str = ".wav", prefix: str = "armtts_") -> Path:
    """Generate a temporary file path."""
    return Path(tempfile.mktemp(suffix=suffix, prefix=prefix))


# ============================================================================
# Duration matching / time-stretching
# ============================================================================

def time_stretch_audio(
    input_path: str | Path,
    output_path: str | Path,
    target_duration: float,
    method: str = "rubberband",
) -> Path:
    """Time-stretch audio to match target duration.

    Args:
        input_path: Source audio file.
        output_path: Destination audio file.
        target_duration: Target duration in seconds.
        method: "rubberband" (high quality) or "ffmpeg" (faster).

    Returns:
        Path to stretched audio file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_duration = get_audio_duration(input_path)
    if source_duration == 0:
        raise ValueError(f"Source audio has zero duration: {input_path}")

    ratio = target_duration / source_duration
    ratio = max(0.5, min(2.0, ratio))  # Safety clamp

    if abs(ratio - 1.0) < 0.02:
        # Less than 2% difference — just copy
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path

    if method == "rubberband":
        cmd = [
            "rubberband",
            "--time", str(ratio),
            "--pitch", "1.0",  # Preserve pitch
            "--crisp", "5",  # High quality
            str(input_path),
            str(output_path),
        ]
    else:
        # atempo is a speed factor (input/output duration), the inverse of
        # rubberband's --time ratio. Chain filters for values outside 0.5-2.0.
        filters = []
        remaining = 1.0 / ratio
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.6f}")

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-filter:a", ",".join(filters),
            str(output_path),
        ]

    subprocess.run(cmd, capture_output=True, check=True)
    logger.debug(
        "Time-stretched {:.1f}s -> {:.1f}s (ratio={:.2f})",
        source_duration, target_duration, ratio,
    )
    return output_path


# ============================================================================
# Consent / Ethics
# ============================================================================

def log_voice_consent(
    speaker_id: str,
    consent_given: bool,
    consent_log: str | Path = "logs/voice_consent.json",
) -> None:
    """Log voice cloning consent for ethics compliance."""
    consent_log = Path(consent_log)
    consent_log.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    if consent_log.exists():
        with open(consent_log) as f:
            entries = json.load(f)

    entries.append({
        "speaker_id": speaker_id,
        "consent_given": consent_given,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    with open(consent_log, "w") as f:
        json.dump(entries, f, indent=2)

    logger.info("Voice consent logged for speaker '{}': {}", speaker_id, consent_given)
