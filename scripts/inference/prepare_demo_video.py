#!/usr/bin/env python3
"""Prepare a short demo video for Colab-friendly dubbing tests.

Two modes are supported:
1. generate: create a synthetic spoken-English demo video with no copyright risk
2. trim: trim an existing local video down to a short Colab-friendly clip

Examples:
    python scripts/inference/prepare_demo_video.py \
        --mode generate \
        --output /content/input_short.mp4

    python scripts/inference/prepare_demo_video.py \
        --mode trim \
        --input /content/my_video.mp4 \
        --duration 15 \
        --output /content/input_short.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import tempfile
from pathlib import Path

from loguru import logger


DEFAULT_TEXT = (
    "Hello. This is a short English test video for Armenian dubbing. "
    "I am speaking clearly so the pipeline can transcribe, translate, "
    "and synthesize the result in Armenian."
)


def run_command(command: list[str]) -> None:
    """Run a subprocess and raise a helpful error on failure."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Command failed: {' '.join(shlex.quote(part) for part in command)}\n{stderr}")


def probe_duration(path: str | Path) -> float:
    """Read media duration with ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


async def synthesize_demo_audio(text: str, output_mp3: Path, voice: str) -> None:
    """Generate spoken demo audio with edge-tts."""
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError(
            "edge-tts is not installed. Install requirements-colab.txt before generating a demo clip."
        ) from exc

    communicator = edge_tts.Communicate(text=text, voice=voice)
    await communicator.save(str(output_mp3))


def generate_demo_video(output_path: Path, text: str, voice: str, fps: int) -> Path:
    """Create a synthetic spoken video with a simple slate background."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="armtts_demo_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        speech_mp3 = temp_dir_path / "speech.mp3"

        asyncio.run(synthesize_demo_audio(text=text, output_mp3=speech_mp3, voice=voice))
        duration = probe_duration(speech_mp3)

        video_filter = (
            "drawtext=text='Armenian Video Dubbing Demo':"
            "fontcolor=white:fontsize=40:"
            "x=(w-text_w)/2:y=(h-text_h)/2"
        )

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=#101820:s=1280x720:r={fps}:d={duration:.2f}",
            "-i",
            str(speech_mp3),
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]

        try:
            run_command(command)
        except RuntimeError:
            fallback_command = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=#101820:s=1280x720:r={fps}:d={duration:.2f}",
                "-i",
                str(speech_mp3),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ]
            run_command(fallback_command)

    logger.info("Generated synthetic demo video: {}", output_path)
    return output_path


def trim_existing_video(
    input_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
    fps: int,
) -> Path:
    """Trim an existing video to a short Colab-friendly clip."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-vf",
        f"fps={fps},scale=-2:720",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-ar",
        "24000",
        str(output_path),
    ]
    run_command(command)
    logger.info("Prepared short input clip: {}", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a short demo video for Colab testing")
    parser.add_argument("--mode", choices=["generate", "trim"], default="generate")
    parser.add_argument("--input", type=str, default=None, help="Existing input video path for trim mode")
    parser.add_argument("--output", type=str, default="input_short.mp4", help="Output video path")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Speech text for generated mode")
    parser.add_argument("--voice", type=str, default="en-US-AriaNeural", help="edge-tts voice for generated mode")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds for trim mode")
    parser.add_argument("--duration", type=float, default=15.0, help="Trim duration in seconds")
    parser.add_argument("--fps", type=int, default=25, help="Output frame rate")

    args = parser.parse_args()
    output_path = Path(args.output)

    if args.mode == "generate":
        generate_demo_video(
            output_path=output_path,
            text=args.text,
            voice=args.voice,
            fps=args.fps,
        )
        return

    if not args.input:
        raise ValueError("--input is required when --mode trim")

    trim_existing_video(
        input_path=Path(args.input),
        output_path=output_path,
        start_time=args.start,
        duration=args.duration,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()