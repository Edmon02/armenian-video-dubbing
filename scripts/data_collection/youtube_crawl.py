#!/usr/bin/env python3
"""
YouTube Armenian Content Crawler — Phase 1a

Replicates the "Scaling Armenian ASR" pipeline (Hakobyan et al., CSIT 2025):
  1. Search YouTube for Armenian-language content across diverse domains
  2. Download audio (best quality) with metadata
  3. Segment into utterance-level chunks using VAD
  4. Filter by language confidence, SNR, and duration

Target: 5,000–8,000+ hours of Armenian speech data.

Usage:
    python scripts/data_collection/youtube_crawl.py --config configs/crawl_config.yaml
    python scripts/data_collection/youtube_crawl.py --phase search   # Only search
    python scripts/data_collection/youtube_crawl.py --phase download # Only download
    python scripts/data_collection/youtube_crawl.py --phase segment  # Only segment
    python scripts/data_collection/youtube_crawl.py --phase all      # Full pipeline
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CRAWL_CONFIG = {
    "output_dir": "data/youtube_crawl",
    "search": {
        "max_results_per_query": 200,
        "max_total_videos": 100000,
        "min_duration_sec": 60,
        "max_duration_sec": 36000,  # 10 hours
        "upload_date_after": "20150101",
        "languages": ["hy"],
        "regions": ["AM", "RU", "US", "FR", "GE", "LB", "IR"],
        "queries": [
            # News & Public affairs
            "հայկական նորություններ",  # Armenian news
            "Հանրային հեռուստատեսություն",  # Public television
            "Ազատություն Ալիք Մեդիա Արմենիա",
            "1TV Armenia",
            "Shant TV",
            "Armenia TV նորություններ",
            "Հայկական հեռուստատեսություն",
            # Podcasts & Talk shows
            "հայերենի փոդքասթ",
            "Armenian podcast",
            "հայերեն հարցազրույց",
            "հայերեն զրույց",
            "հարց ու պատասխան",
            # Education & Lectures
            "Armenian lecture university",
            "Yerevan State University lecture",
            "AUA lecture Armenian",
            "Armenian TED talk",
            "գիտություն և տեխնոլոգիա",
            # Audiobooks & Literature
            "հայերեն աուդիոգիրք",
            "Armenian audiobook",
            "Hovhannes Tumanyan",
            "William Saroyan Armenian",
            # Religion & Sermons
            "Հայաստանյայց եկեղեցի քարոզ",
            "Armenian church sermon",
            "Armenian liturgy",
            # Vlogs & Lifestyle
            "Armenian vlogger",
            "Yerevan vlog",
            "հայերեն vlog",
            "Armenia travel Armenian language",
            # History & Documentary
            "Armenian history documentary",
            "հայերեն վավերագրական ֆիլմ",
            "Հայոց պատմություն",
            # Cooking & Culture
            "Armenian cooking recipe",
            "հայկական խոհանոց",
            # Technology & Science
            "technology Armenian",
            "IT Armenia Armenian",
            "coding Armenian tutorial",
            # Music discussions (not music itself)
            "Armenian music review",
            "Armenian artist interview",
            # Sports
            "Armenian football commentary",
            "Armenian sport news",
            # Children
            "Armenian children story",
            "հայերեն մանկական պատմություն",
            # Politics & Analysis
            "Armenian political analysis",
            "Armenian parliament session",
            # Diaspora content
            "Armenian diaspora interview",
            "Western Armenian conversation",
            "Los Angeles Armenian",
            "Beirut Armenian",
        ],
        # Channel IDs with known high-quality Armenian speech
        "channel_ids": [
            # Add known Armenian content creator channel IDs here
        ],
        "playlist_ids": [
            # Add known playlist IDs here
        ],
    },
    "download": {
        "format": "bestaudio/best",
        "audio_format": "wav",
        "audio_quality": 0,
        "sample_rate": 16000,
        "max_concurrent": 4,
        "retry_attempts": 3,
        "cookies_file": None,  # Path to cookies.txt for age-restricted content
        "rate_limit": "5M",  # Download speed limit
        "sleep_interval": 2,  # Seconds between downloads
    },
    "segment": {
        "vad_aggressiveness": 2,  # 0-3, higher = more aggressive
        "min_segment_sec": 1.0,
        "max_segment_sec": 30.0,
        "target_segment_sec": 15.0,
        "min_speech_ratio": 0.6,  # Min ratio of speech in segment
        "padding_ms": 300,
        "sample_rate": 16000,
    },
    "filter": {
        "min_snr_db": 10,
        "min_duration_sec": 1.0,
        "max_duration_sec": 30.0,
        "language_confidence_threshold": 0.7,
    },
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class VideoMeta:
    video_id: str
    title: str = ""
    channel: str = ""
    duration_sec: float = 0
    upload_date: str = ""
    description: str = ""
    url: str = ""
    language: str = "hy"
    downloaded: bool = False
    audio_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================
# Phase 1: YouTube Search
# ============================================================================

class YouTubeSearcher:
    """Search YouTube for Armenian-language content using yt-dlp."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.metadata_file = output_dir / "video_metadata.jsonl"
        self.seen_ids: set[str] = set()
        self._load_existing()

    def _load_existing(self):
        """Load already-discovered video IDs."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                for line in f:
                    try:
                        meta = json.loads(line.strip())
                        self.seen_ids.add(meta["video_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            logger.info("Loaded {} existing video IDs", len(self.seen_ids))

    def search_query(self, query: str, max_results: int = 200) -> list[VideoMeta]:
        """Search YouTube with a single query."""
        logger.info("Searching: '{}' (max {})", query, max_results)

        cmd = [
            "yt-dlp",
            f"ytsearch{max_results}:{query}",
            "--flat-playlist",
            "--dump-json",
            "--no-download",
            "--ignore-errors",
            "--no-warnings",
        ]

        results = []
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )

            for line in proc.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    info = json.loads(line)
                    vid_id = info.get("id", "")

                    if not vid_id or vid_id in self.seen_ids:
                        continue

                    duration = info.get("duration") or 0

                    # Filter by duration
                    if duration < self.config["min_duration_sec"]:
                        continue
                    if duration > self.config["max_duration_sec"]:
                        continue

                    meta = VideoMeta(
                        video_id=vid_id,
                        title=info.get("title", ""),
                        channel=info.get("channel", info.get("uploader", "")),
                        duration_sec=duration,
                        upload_date=info.get("upload_date", ""),
                        description=info.get("description", "")[:500],
                        url=f"https://www.youtube.com/watch?v={vid_id}",
                    )
                    results.append(meta)
                    self.seen_ids.add(vid_id)

                except (json.JSONDecodeError, KeyError):
                    continue

        except subprocess.TimeoutExpired:
            logger.warning("Search timed out for query: {}", query)
        except Exception as e:
            logger.error("Search error for '{}': {}", query, e)

        logger.info("  Found {} new videos for '{}'", len(results), query)
        return results

    def search_channel(self, channel_id: str, max_results: int = 500) -> list[VideoMeta]:
        """Get all videos from a channel."""
        logger.info("Crawling channel: {}", channel_id)

        cmd = [
            "yt-dlp",
            f"https://www.youtube.com/channel/{channel_id}/videos",
            "--flat-playlist",
            "--dump-json",
            "--no-download",
            "--ignore-errors",
            "--playlist-end", str(max_results),
        ]

        results = []
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            for line in proc.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    info = json.loads(line)
                    vid_id = info.get("id", "")
                    if not vid_id or vid_id in self.seen_ids:
                        continue
                    meta = VideoMeta(
                        video_id=vid_id,
                        title=info.get("title", ""),
                        channel=info.get("channel", ""),
                        duration_sec=info.get("duration") or 0,
                        upload_date=info.get("upload_date", ""),
                        url=f"https://www.youtube.com/watch?v={vid_id}",
                    )
                    results.append(meta)
                    self.seen_ids.add(vid_id)
                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception as e:
            logger.error("Channel crawl error: {}", e)

        return results

    def run(self) -> int:
        """Execute full search phase. Returns total videos found."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        total_new = 0
        max_total = self.config.get("max_total_videos", 100000)

        # Search by queries
        for query in self.config.get("queries", []):
            if len(self.seen_ids) >= max_total:
                logger.info("Reached max total videos ({})", max_total)
                break

            results = self.search_query(
                query,
                self.config.get("max_results_per_query", 200),
            )

            # Append to metadata file
            with open(self.metadata_file, "a") as f:
                for meta in results:
                    f.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")

            total_new += len(results)
            time.sleep(1)  # Rate limiting

        # Search by channels
        for ch_id in self.config.get("channel_ids", []):
            if len(self.seen_ids) >= max_total:
                break

            results = self.search_channel(ch_id)
            with open(self.metadata_file, "a") as f:
                for meta in results:
                    f.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")
            total_new += len(results)

        logger.info("Search complete: {} new videos, {} total", total_new, len(self.seen_ids))
        return total_new


# ============================================================================
# Phase 2: Download Audio
# ============================================================================

class AudioDownloader:
    """Download audio from YouTube videos using yt-dlp."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.audio_dir = output_dir / "raw_audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def download_single(self, meta: VideoMeta) -> VideoMeta:
        """Download audio for a single video."""
        output_path = self.audio_dir / f"{meta.video_id}.wav"

        if output_path.exists():
            meta.downloaded = True
            meta.audio_path = str(output_path)
            return meta

        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", f"ffmpeg:-ar {self.config.get('sample_rate', 16000)} -ac 1",
            "--output", str(self.audio_dir / "%(id)s.%(ext)s"),
            "--no-playlist",
            "--ignore-errors",
            "--retries", str(self.config.get("retry_attempts", 3)),
        ]

        if self.config.get("rate_limit"):
            cmd.extend(["--limit-rate", self.config["rate_limit"]])

        if self.config.get("cookies_file"):
            cmd.extend(["--cookies", self.config["cookies_file"]])

        cmd.append(meta.url)

        try:
            subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, check=True
            )

            if output_path.exists():
                meta.downloaded = True
                meta.audio_path = str(output_path)
                logger.debug("Downloaded: {} ({:.0f}s)", meta.video_id, meta.duration_sec)
            else:
                # yt-dlp might save with different extension; find it
                candidates = list(self.audio_dir.glob(f"{meta.video_id}.*"))
                if candidates:
                    # Convert to wav if needed
                    src = candidates[0]
                    if src.suffix != ".wav":
                        subprocess.run([
                            "ffmpeg", "-y", "-i", str(src),
                            "-ar", str(self.config.get("sample_rate", 16000)),
                            "-ac", "1",
                            str(output_path),
                        ], capture_output=True, check=True)
                        src.unlink()
                    meta.downloaded = True
                    meta.audio_path = str(output_path)
                else:
                    meta.error = "File not found after download"
        except subprocess.TimeoutExpired:
            meta.error = "Download timeout"
        except subprocess.CalledProcessError as e:
            meta.error = f"Download failed: {e.stderr[:200] if e.stderr else 'unknown'}"
        except Exception as e:
            meta.error = str(e)

        return meta

    def run(self) -> tuple[int, int]:
        """Download all pending videos. Returns (success, failed) counts."""
        metadata_file = self.output_dir / "video_metadata.jsonl"
        if not metadata_file.exists():
            logger.error("No metadata file found. Run search phase first.")
            return 0, 0

        # Load all metadata
        videos = []
        with open(metadata_file) as f:
            for line in f:
                try:
                    meta = VideoMeta.from_dict(json.loads(line.strip()))
                    if not meta.downloaded and not meta.error:
                        videos.append(meta)
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info("Downloading audio for {} videos...", len(videos))

        success = 0
        failed = 0
        max_concurrent = self.config.get("max_concurrent", 4)
        sleep_interval = self.config.get("sleep_interval", 2)

        # Track results for re-writing metadata
        results = []

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {}
            for meta in videos:
                future = executor.submit(self.download_single, meta)
                futures[future] = meta

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.downloaded:
                    success += 1
                else:
                    failed += 1

                if (success + failed) % 50 == 0:
                    logger.info(
                        "Progress: {}/{} downloaded, {} failed",
                        success, len(videos), failed,
                    )

                time.sleep(sleep_interval)

        # Update metadata file
        updated_file = self.output_dir / "video_metadata_updated.jsonl"
        with open(updated_file, "w") as f:
            for meta in results:
                f.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")

        # Also preserve original unprocessed entries
        with open(metadata_file) as f:
            existing = {}
            for line in f:
                try:
                    d = json.loads(line.strip())
                    existing[d["video_id"]] = d
                except (json.JSONDecodeError, KeyError):
                    continue

        for meta in results:
            existing[meta.video_id] = meta.to_dict()

        with open(metadata_file, "w") as f:
            for d in existing.values():
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        logger.info("Download complete: {} success, {} failed", success, failed)
        return success, failed


# ============================================================================
# Phase 3: VAD Segmentation
# ============================================================================

class VADSegmenter:
    """Segment long audio files into utterance-level chunks using WebRTC VAD."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.segments_dir = output_dir / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.sr = config.get("sample_rate", 16000)

    def _read_wav_frames(self, path: Path) -> bytes:
        """Read wav file as raw PCM bytes at target sample rate."""
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(str(path), dtype="int16")

        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype("int16")

        # Resample if needed
        if sr != self.sr:
            import librosa
            audio_f = audio.astype("float32") / 32768.0
            audio_f = librosa.resample(audio_f, orig_sr=sr, target_sr=self.sr)
            audio = (audio_f * 32768).astype("int16")

        return audio.tobytes()

    def _vad_segments(self, audio_bytes: bytes) -> list[tuple[float, float]]:
        """Run VAD and return list of (start_sec, end_sec) speech segments."""
        import webrtcvad

        vad = webrtcvad.Vad(self.config.get("vad_aggressiveness", 2))
        frame_duration_ms = 30  # WebRTC VAD supports 10, 20, 30 ms
        frame_size = int(self.sr * frame_duration_ms / 1000) * 2  # 2 bytes per sample

        # Split into frames
        frames = []
        for i in range(0, len(audio_bytes) - frame_size, frame_size):
            frame = audio_bytes[i:i + frame_size]
            if len(frame) == frame_size:
                frames.append(frame)

        if not frames:
            return []

        # Get speech/silence for each frame
        is_speech = []
        for frame in frames:
            try:
                is_speech.append(vad.is_speech(frame, self.sr))
            except Exception:
                is_speech.append(False)

        # Merge consecutive speech frames into segments
        padding_frames = int(self.config.get("padding_ms", 300) / frame_duration_ms)
        min_frames = int(self.config.get("min_segment_sec", 1.0) * 1000 / frame_duration_ms)
        max_frames = int(self.config.get("max_segment_sec", 30.0) * 1000 / frame_duration_ms)

        segments = []
        in_speech = False
        start_frame = 0
        silence_count = 0
        max_silence = int(500 / frame_duration_ms)  # 500ms silence triggers split

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    start_frame = max(0, i - padding_frames)
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count > max_silence:
                        end_frame = min(len(is_speech), i + padding_frames)
                        seg_len = end_frame - start_frame

                        if seg_len >= min_frames:
                            # Split if too long
                            if seg_len > max_frames:
                                for chunk_start in range(start_frame, end_frame, max_frames):
                                    chunk_end = min(chunk_start + max_frames, end_frame)
                                    if chunk_end - chunk_start >= min_frames:
                                        segments.append((
                                            chunk_start * frame_duration_ms / 1000,
                                            chunk_end * frame_duration_ms / 1000,
                                        ))
                            else:
                                segments.append((
                                    start_frame * frame_duration_ms / 1000,
                                    end_frame * frame_duration_ms / 1000,
                                ))

                        in_speech = False
                        silence_count = 0

        # Handle final segment
        if in_speech:
            end_frame = len(is_speech)
            seg_len = end_frame - start_frame
            if seg_len >= min_frames:
                segments.append((
                    start_frame * frame_duration_ms / 1000,
                    end_frame * frame_duration_ms / 1000,
                ))

        return segments

    def segment_file(self, audio_path: Path, video_id: str) -> list[dict]:
        """Segment a single audio file. Returns list of segment metadata."""
        import soundfile as sf
        import numpy as np

        try:
            audio_bytes = self._read_wav_frames(audio_path)
        except Exception as e:
            logger.error("Failed to read {}: {}", audio_path, e)
            return []

        segments_meta = self._vad_segments(audio_bytes)

        if not segments_meta:
            logger.debug("No speech segments found in {}", video_id)
            return []

        # Load audio as float for saving segments
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        results = []
        video_seg_dir = self.segments_dir / video_id
        video_seg_dir.mkdir(parents=True, exist_ok=True)

        for idx, (start, end) in enumerate(segments_meta):
            start_sample = int(start * self.sr)
            end_sample = int(end * self.sr)

            segment_audio = audio[start_sample:end_sample]
            duration = len(segment_audio) / self.sr

            # Skip very short or very long
            if duration < self.config.get("min_segment_sec", 1.0):
                continue
            if duration > self.config.get("max_segment_sec", 30.0):
                continue

            # Compute SNR estimate
            rms = np.sqrt(np.mean(segment_audio ** 2))
            if rms < 1e-7:
                continue

            seg_filename = f"{video_id}_{idx:05d}.wav"
            seg_path = video_seg_dir / seg_filename

            sf.write(str(seg_path), segment_audio, self.sr)

            results.append({
                "segment_id": f"{video_id}_{idx:05d}",
                "video_id": video_id,
                "audio_path": str(seg_path),
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(duration, 3),
                "rms": round(float(rms), 6),
            })

        return results

    def run(self) -> int:
        """Segment all downloaded audio files. Returns total segments."""
        from tqdm import tqdm

        audio_dir = self.output_dir / "raw_audio"
        if not audio_dir.exists():
            logger.error("No raw_audio directory. Run download phase first.")
            return 0

        audio_files = sorted(audio_dir.glob("*.wav"))
        logger.info("Segmenting {} audio files...", len(audio_files))

        manifest_path = self.output_dir / "segments_manifest.jsonl"
        total_segments = 0
        total_duration = 0.0

        with open(manifest_path, "w") as f:
            for audio_file in tqdm(audio_files, desc="Segmenting"):
                video_id = audio_file.stem
                segments = self.segment_file(audio_file, video_id)

                for seg in segments:
                    f.write(json.dumps(seg, ensure_ascii=False) + "\n")
                    total_segments += 1
                    total_duration += seg["duration_sec"]

        hours = total_duration / 3600
        logger.info(
            "Segmentation complete: {} segments, {:.1f} hours from {} files",
            total_segments, hours, len(audio_files),
        )
        return total_segments


# ============================================================================
# SNR Filter
# ============================================================================

class SNRFilter:
    """Filter segments based on signal-to-noise ratio estimation."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.min_snr = config.get("min_snr_db", 10)

    @staticmethod
    def estimate_snr(audio_path: str) -> float:
        """Estimate SNR using a simple energy-based method."""
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Frame-level energy
        frame_size = int(0.025 * sr)  # 25ms frames
        hop = int(0.010 * sr)  # 10ms hop

        energies = []
        for i in range(0, len(audio) - frame_size, hop):
            frame = audio[i:i + frame_size]
            energies.append(np.mean(frame ** 2))

        if not energies:
            return 0.0

        energies = np.array(energies)
        energies = energies[energies > 0]

        if len(energies) < 4:
            return 0.0

        # Sort energies; top 80% = signal, bottom 20% = noise
        sorted_e = np.sort(energies)
        n = len(sorted_e)
        noise_e = np.mean(sorted_e[:max(1, n // 5)])
        signal_e = np.mean(sorted_e[n // 5:])

        if noise_e <= 0:
            return 60.0  # Very clean

        snr_db = 10 * np.log10(signal_e / noise_e)
        return float(snr_db)

    def run(self) -> tuple[int, int]:
        """Filter segments manifest by SNR. Returns (kept, removed)."""
        from tqdm import tqdm

        manifest = self.output_dir / "segments_manifest.jsonl"
        if not manifest.exists():
            logger.error("No segments manifest. Run segment phase first.")
            return 0, 0

        segments = []
        with open(manifest) as f:
            for line in f:
                try:
                    segments.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        logger.info("Filtering {} segments by SNR (min {}dB)...", len(segments), self.min_snr)

        kept = []
        removed = 0

        for seg in tqdm(segments, desc="SNR filtering"):
            try:
                snr = self.estimate_snr(seg["audio_path"])
                seg["snr_db"] = round(snr, 1)
                if snr >= self.min_snr:
                    kept.append(seg)
                else:
                    removed += 1
            except Exception:
                removed += 1

        # Write filtered manifest
        filtered_manifest = self.output_dir / "segments_filtered.jsonl"
        with open(filtered_manifest, "w") as f:
            for seg in kept:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

        total_hours = sum(s["duration_sec"] for s in kept) / 3600
        logger.info(
            "SNR filter: kept {} ({:.1f}h), removed {}",
            len(kept), total_hours, removed,
        )
        return len(kept), removed


# ============================================================================
# CLI Entry Point
# ============================================================================

def load_crawl_config(config_path: Optional[str] = None) -> dict:
    """Load crawl config from YAML or use defaults."""
    config = DEFAULT_CRAWL_CONFIG.copy()

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        # Deep merge
        for key in user_config:
            if key in config and isinstance(config[key], dict):
                config[key].update(user_config[key])
            else:
                config[key] = user_config[key]

    return config


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Armenian Content Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["search", "download", "segment", "filter", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to crawl config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    args = parser.parse_args()

    setup_logger()
    config = load_crawl_config(args.config)

    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config
    with open(output_dir / "crawl_config_effective.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info("YouTube Crawl Pipeline — Output: {}", output_dir)

    if args.phase in ("search", "all"):
        logger.info("=" * 50)
        logger.info("PHASE: Search")
        logger.info("=" * 50)
        searcher = YouTubeSearcher(config["search"], output_dir)
        searcher.run()

    if args.phase in ("download", "all"):
        logger.info("=" * 50)
        logger.info("PHASE: Download")
        logger.info("=" * 50)
        downloader = AudioDownloader(config["download"], output_dir)
        downloader.run()

    if args.phase in ("segment", "all"):
        logger.info("=" * 50)
        logger.info("PHASE: Segment")
        logger.info("=" * 50)
        segmenter = VADSegmenter(config["segment"], output_dir)
        segmenter.run()

    if args.phase in ("filter", "all"):
        logger.info("=" * 50)
        logger.info("PHASE: SNR Filter")
        logger.info("=" * 50)
        snr_filter = SNRFilter(config["filter"], output_dir)
        snr_filter.run()

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
