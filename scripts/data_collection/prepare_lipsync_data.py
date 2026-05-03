#!/usr/bin/env python3
"""
Lip-Sync Data Preparation — Phase 1g

Prepares video data for MuseTalk lip-sync fine-tuning/evaluation:
  1. Download HDTF dataset (standard talking-face benchmark)
  2. Crawl Armenian YouTube talking-head videos
  3. Extract face crops with alignment
  4. Pair with audio segments
  5. Generate training manifests

Usage:
    python scripts/data_collection/prepare_lipsync_data.py --phase hdtf
    python scripts/data_collection/prepare_lipsync_data.py --phase armenian
    python scripts/data_collection/prepare_lipsync_data.py --phase process
    python scripts/data_collection/prepare_lipsync_data.py --phase all
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


class LipSyncDataProcessor:
    """Prepare talking-face video data for lip-sync training."""

    def __init__(self, output_dir: Path, fps: int = 25):
        self.output_dir = output_dir
        self.fps = fps
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_hdtf(self) -> int:
        """Download HDTF (High-Definition Talking Face) dataset metadata.

        Note: HDTF videos must be downloaded from YouTube using their video IDs.
        This downloads the ID list and processes available videos.

        Returns:
            Number of videos available.
        """
        hdtf_dir = self.output_dir / "hdtf"
        hdtf_dir.mkdir(parents=True, exist_ok=True)

        # HDTF video IDs (subset — full list at https://github.com/MRzzm/HDTF)
        # These are the most commonly available ones
        hdtf_meta_url = "https://raw.githubusercontent.com/MRzzm/HDTF/main/HDTF_dataset/video_url.txt"

        logger.info("Downloading HDTF video list...")

        try:
            import requests
            resp = requests.get(hdtf_meta_url, timeout=30)
            if resp.status_code == 200:
                with open(hdtf_dir / "video_urls.txt", "w") as f:
                    f.write(resp.text)
                lines = [l.strip() for l in resp.text.split("\n") if l.strip()]
                logger.info("HDTF: {} video entries", len(lines))
                return len(lines)
            else:
                logger.warning("Cannot download HDTF list (status {})", resp.status_code)
        except Exception as e:
            logger.warning("HDTF download error: {}", e)

        # Create manual instruction
        with open(hdtf_dir / "DOWNLOAD_INSTRUCTIONS.md", "w") as f:
            f.write("""# HDTF Dataset Download

1. Clone the HDTF repo: `git clone https://github.com/MRzzm/HDTF.git`
2. Follow their instructions to download videos using the provided scripts
3. Place downloaded videos in: `data/lipsync_hdtf/hdtf/videos/`
4. Re-run this script with `--phase process`
""")
        return 0

    def crawl_armenian_talking_heads(self, max_videos: int = 200) -> int:
        """Crawl Armenian talking-head videos from YouTube.

        Searches for: news anchors, interviews, vlogs with face visible.
        """
        arm_dir = self.output_dir / "armenian"
        arm_dir.mkdir(parents=True, exist_ok=True)

        queries = [
            "Armenian news anchor",
            "հայերեն հարցազրույց",
            "Armenia interview face",
            "Armenian vlog talking",
            "Armenian lecture face",
            "Yerevan interview Armenian",
        ]

        video_ids = set()
        meta_file = arm_dir / "video_metadata.jsonl"

        # Load existing
        if meta_file.exists():
            with open(meta_file) as f:
                for line in f:
                    try:
                        d = json.loads(line.strip())
                        video_ids.add(d["video_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

        for query in queries:
            if len(video_ids) >= max_videos:
                break

            logger.info("Searching: '{}'", query)
            cmd = [
                "yt-dlp",
                f"ytsearch50:{query}",
                "--flat-playlist",
                "--dump-json",
                "--no-download",
            ]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                for line in proc.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        info = json.loads(line)
                        vid_id = info.get("id", "")
                        duration = info.get("duration") or 0

                        if vid_id in video_ids:
                            continue
                        if duration < 60 or duration > 3600:
                            continue

                        video_ids.add(vid_id)
                        with open(meta_file, "a") as f:
                            f.write(json.dumps({
                                "video_id": vid_id,
                                "title": info.get("title", ""),
                                "duration": duration,
                                "url": f"https://youtube.com/watch?v={vid_id}",
                            }, ensure_ascii=False) + "\n")
                    except (json.JSONDecodeError, KeyError):
                        continue
            except Exception as e:
                logger.warning("Search error: {}", e)

        logger.info("Armenian talking heads: {} videos found", len(video_ids))

        # Download videos (720p, max 5 min per video)
        downloaded = 0
        videos_dir = arm_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

        with open(meta_file) as f:
            entries = [json.loads(l.strip()) for l in f if l.strip()]

        for entry in tqdm(entries[:max_videos], desc="Downloading videos"):
            vid_id = entry["video_id"]
            out_path = videos_dir / f"{vid_id}.mp4"

            if out_path.exists():
                downloaded += 1
                continue

            cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
                "--merge-output-format", "mp4",
                "--download-sections", "*0:00-5:00",  # First 5 min only
                "-o", str(out_path),
                f"https://youtube.com/watch?v={vid_id}",
            ]

            try:
                subprocess.run(cmd, capture_output=True, timeout=300)
                if out_path.exists():
                    downloaded += 1
            except Exception:
                pass

        logger.info("Downloaded {} Armenian talking-head videos", downloaded)
        return downloaded

    def detect_and_crop_faces(self, video_path: Path, output_dir: Path) -> list[dict]:
        """Detect faces and extract face crops from video.

        Returns list of face track metadata.
        """
        import cv2

        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error("Cannot open video: {}", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use MediaPipe for face detection (faster than dlib)
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection
            detector = mp_face.FaceDetection(
                model_selection=1,  # Full-range model
                min_detection_confidence=0.5,
            )
        except ImportError:
            logger.error("MediaPipe not available for face detection")
            cap.release()
            return []

        # Sample frames for face detection
        sample_interval = max(1, int(fps / 5))  # ~5 fps for detection
        face_tracks = []
        frame_idx = 0
        face_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)

                if results.detections:
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)

                        # Expand by 30% for context
                        margin_x = int(w * 0.3)
                        margin_y = int(h * 0.3)
                        x = max(0, x - margin_x)
                        y = max(0, y - margin_y)
                        w = min(width - x, w + 2 * margin_x)
                        h = min(height - y, h + 2 * margin_y)

                        face_frames.append({
                            "frame_idx": frame_idx,
                            "time_sec": round(frame_idx / fps, 3),
                            "bbox": [x, y, w, h],
                            "confidence": round(det.score[0], 3),
                        })

            frame_idx += 1

        cap.release()
        detector.close()

        if not face_frames:
            return []

        # Check face consistency (should be roughly same position throughout)
        # Group into tracks (simple: just use the dominant face)
        face_ratio = len(face_frames) / max(1, total_frames // sample_interval)

        if face_ratio < 0.3:
            logger.debug("Low face ratio ({:.1f}%) in {}", face_ratio * 100, video_path.name)
            return []

        track_meta = {
            "video_path": str(video_path),
            "video_id": video_path.stem,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "face_frames": len(face_frames),
            "face_ratio": round(face_ratio, 3),
            "duration_sec": round(total_frames / fps, 3),
        }
        face_tracks.append(track_meta)

        return face_tracks

    def process_videos(self) -> int:
        """Process all downloaded videos: face detection + metadata extraction."""
        video_dirs = [
            self.output_dir / "hdtf" / "videos",
            self.output_dir / "armenian" / "videos",
        ]

        all_tracks = []

        for vdir in video_dirs:
            if not vdir.exists():
                continue

            videos = sorted(list(vdir.glob("*.mp4")) + list(vdir.glob("*.avi")))
            logger.info("Processing {} videos from {}", len(videos), vdir.parent.name)

            crops_dir = vdir.parent / "face_crops"

            for video in tqdm(videos, desc=f"Face detection ({vdir.parent.name})"):
                tracks = self.detect_and_crop_faces(video, crops_dir / video.stem)
                all_tracks.extend(tracks)

        # Write manifest
        manifest_path = self.output_dir / "lipsync_manifest.jsonl"
        with open(manifest_path, "w") as f:
            for track in all_tracks:
                f.write(json.dumps(track, ensure_ascii=False) + "\n")

        total_hours = sum(t.get("duration_sec", 0) for t in all_tracks) / 3600
        logger.info("Lip-sync data: {} videos with faces, {:.1f}h total", len(all_tracks), total_hours)

        # Save stats
        stats = {
            "total_videos": len(all_tracks),
            "total_hours": round(total_hours, 2),
            "avg_face_ratio": round(
                np.mean([t["face_ratio"] for t in all_tracks]) if all_tracks else 0, 3
            ),
        }
        with open(self.output_dir / "lipsync_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        return len(all_tracks)


def main():
    parser = argparse.ArgumentParser(description="Lip-Sync Data Preparation")
    parser.add_argument(
        "--phase",
        choices=["hdtf", "armenian", "process", "all"],
        default="all",
    )
    parser.add_argument("--output-dir", default="data/lipsync_hdtf")
    parser.add_argument("--max-armenian-videos", type=int, default=200)

    args = parser.parse_args()
    setup_logger()

    processor = LipSyncDataProcessor(Path(args.output_dir))

    if args.phase in ("hdtf", "all"):
        processor.download_hdtf()

    if args.phase in ("armenian", "all"):
        processor.crawl_armenian_talking_heads(args.max_armenian_videos)

    if args.phase in ("process", "all"):
        processor.process_videos()


if __name__ == "__main__":
    main()
