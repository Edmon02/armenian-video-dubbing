#!/usr/bin/env python3
"""
Lip-Sync Metrics (LSE-C/D) for Video Dubbing Quality Assessment.

Metrics:
- LSE-C: Lip Sync Error Confidence (audio-to-visual synchronization)
- LSE-D: Lip Sync Error Distance (temporal offset measurement)
- Per-frame sync errors
- Failure detection and analysis
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import json

import numpy as np
import torch
from loguru import logger

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available (optional)")

from src.utils.helpers import load_audio, timer

# Mouth tracking below is a stub, so LSE-C/LSE-D from this module would be
# random numbers. Use the SyncNet scorer in notebooks/colab_dubbing_ablation.ipynb.
MOUTH_TRACKING_IMPLEMENTED = False
UNIMPLEMENTED_NOTE = (
    "LSE-C/LSE-D not implemented here (mouth tracking is a stub); "
    "use syncnet_python as in notebooks/colab_dubbing_ablation.ipynb"
)


class LipSyncMetricsComputer:
    """Compute lip-sync quality metrics for dubbed videos."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize lip-sync metrics computer.

        Args:
            device: Device to run on
        """
        self.device = device
        self.sample_rate = 44100

        logger.info("LipSyncMetricsComputer initialized")

    def compute_lse_c_metric(
        self,
        video_path: str,
        dubbed_audio_path: str,
    ) -> Dict:
        """
        Compute LSE-C (Lip Sync Error Confidence).

        Measures audio-to-visual synchronization accuracy.

        Args:
            video_path: Path to dubbed video file
            dubbed_audio_path: Path to dubbed audio

        Returns:
            Dictionary with LSE-C score and metadata
        """
        logger.info(f"Computing LSE-C for: {video_path}")

        if not MOUTH_TRACKING_IMPLEMENTED:
            logger.warning(UNIMPLEMENTED_NOTE)
            return {"error": UNIMPLEMENTED_NOTE}

        try:
            # Extract mouth movements from video
            with timer("Mouth movement extraction"):
                mouth_movements = self._extract_mouth_movements(video_path)

            # Extract acoustic features from audio
            with timer("Acoustic feature extraction"):
                acoustic_features = self._extract_acoustic_features(dubbed_audio_path)

            # Compute temporal correlation
            lse_c = self._compute_temporal_correlation(mouth_movements, acoustic_features)

            confidence = self._compute_lse_confidence(mouth_movements, acoustic_features)

            logger.info(f"LSE-C score: {lse_c:.2f}, confidence: {confidence:.2f}")

            return {
                "lse_c": float(lse_c),
                "confidence": float(confidence),
                "n_frames": len(mouth_movements),
            }

        except Exception as e:
            logger.error(f"Failed to compute LSE-C: {e}")
            return {"error": str(e)}

    def compute_lse_d_metric(
        self,
        video_path: str,
        dubbed_audio_path: str,
    ) -> Dict:
        """
        Compute LSE-D (Lip Sync Error Distance).

        Measures temporal offset between lip movements and audio.

        Args:
            video_path: Path to dubbed video
            dubbed_audio_path: Path to dubbed audio

        Returns:
            Dictionary with LSE-D score and offset
        """
        logger.info(f"Computing LSE-D for: {video_path}")

        if not MOUTH_TRACKING_IMPLEMENTED:
            logger.warning(UNIMPLEMENTED_NOTE)
            return {"error": UNIMPLEMENTED_NOTE}

        try:
            # Extract features
            mouth_movements = self._extract_mouth_movements(video_path)
            acoustic_features = self._extract_acoustic_features(dubbed_audio_path)

            # Compute alignment offset
            lse_d, offset_frames = self._compute_alignment_offset(mouth_movements, acoustic_features)

            # Convert frames to milliseconds (assume 25 fps)
            offset_ms = (offset_frames / 25.0) * 1000

            logger.info(f"LSE-D score: {lse_d:.2f}, offset: {offset_ms:.0f} ms")

            return {
                "lse_d": float(lse_d),
                "offset_ms": float(offset_ms),
                "offset_frames": int(offset_frames),
            }

        except Exception as e:
            logger.error(f"Failed to compute LSE-D: {e}")
            return {"error": str(e)}

    def batch_lipsync_evaluation(
        self,
        video_list: list,
        dubbed_audio_list: list,
    ) -> Dict:
        """
        Evaluate lip-sync for multiple videos.

        Args:
            video_list: List of video paths
            dubbed_audio_list: List of dubbed audio paths

        Returns:
            Dictionary with batch lip-sync metrics
        """
        if len(video_list) != len(dubbed_audio_list):
            raise ValueError("Video and audio lists must have same length")

        results = []

        logger.info(f"Evaluating lip-sync for {len(video_list)} videos")

        for video_path, audio_path in zip(video_list, dubbed_audio_list):
            try:
                lse_c_result = self.compute_lse_c_metric(video_path, audio_path)
                lse_d_result = self.compute_lse_d_metric(video_path, audio_path)

                results.append({
                    "video": str(video_path),
                    "lse_c": lse_c_result.get("lse_c"),
                    "lse_d": lse_d_result.get("lse_d"),
                    "offset_ms": lse_d_result.get("offset_ms"),
                })

            except Exception as e:
                logger.error(f"Failed to evaluate {video_path}: {e}")
                results.append({
                    "video": str(video_path),
                    "error": str(e),
                })

        # Aggregate statistics
        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return {"error": "No valid results"}

        lse_c_scores = [r["lse_c"] for r in valid_results if r["lse_c"] is not None]
        lse_d_scores = [r["lse_d"] for r in valid_results if r["lse_d"] is not None]

        return {
            "per_video_results": results,
            "mean_lse_c": float(np.mean(lse_c_scores)) if lse_c_scores else None,
            "mean_lse_d": float(np.mean(lse_d_scores)) if lse_d_scores else None,
            "std_lse_c": float(np.std(lse_c_scores)) if lse_c_scores else None,
            "std_lse_d": float(np.std(lse_d_scores)) if lse_d_scores else None,
            "n_videos": len(results),
            "n_success": len(valid_results),
        }

    def detect_lip_sync_failures(
        self,
        lse_c_scores: list,
        lse_d_scores: list,
        lse_threshold: float = 1.8,
    ) -> Dict:
        """
        Identify videos with poor lip-sync (scores > threshold).

        Args:
            lse_c_scores: List of LSE-C scores
            lse_d_scores: List of LSE-D scores
            lse_threshold: Maximum acceptable LSE score

        Returns:
            Dictionary with failure analysis
        """
        failed_c = [i for i, score in enumerate(lse_c_scores) if score > lse_threshold]
        failed_d = [i for i, score in enumerate(lse_d_scores) if score > lse_threshold]

        failed_indices = set(failed_c) | set(failed_d)

        if not failed_indices:
            return {
                "failures_detected": False,
                "failed_count": 0,
            }

        logger.warning(f"Lip-sync failures: {len(failed_indices)}")

        return {
            "failures_detected": True,
            "failed_count": len(failed_indices),
            "failed_lse_c": [i for i in failed_c],
            "failed_lse_d": [i for i in failed_d],
            "failure_rate": len(failed_indices) / max(1, len(lse_c_scores)),
        }

    # Helper methods

    def _extract_mouth_movements(self, video_path: str) -> np.ndarray:
        """
        Extract mouth movement features from video.

        Uses MediaPipe or simpler CV-based detection.

        Args:
            video_path: Path to video file

        Returns:
            Array of mouth movement features over time
        """
        if not MOUTH_TRACKING_IMPLEMENTED:
            raise NotImplementedError(UNIMPLEMENTED_NOTE)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        mouth_sizes = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Simple mouth detection (mock)
            # In real implementation, use MediaPipe face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mouth_size = np.random.rand()  # Mock

            mouth_sizes.append(mouth_size)

        cap.release()

        return np.array(mouth_sizes)

    def _extract_acoustic_features(self, audio_path: str) -> np.ndarray:
        """
        Extract acoustic features (voicing, energy) from audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Array of acoustic features over time
        """
        from src.utils.helpers import load_audio
        import librosa

        audio, sr = load_audio(audio_path)

        # Compute energy envelope
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        energy = np.sqrt(np.sum(S ** 2, axis=0))

        # Resample to match video frame rate (25 fps)
        n_frames = int(len(audio) / sr * 25)  # 25 fps
        resampled = np.interp(
            np.linspace(0, 1, n_frames),
            np.linspace(0, 1, len(energy)),
            energy
        )

        return resampled / np.max(resampled + 1e-10)

    def _compute_temporal_correlation(
        self,
        mouth_movements: np.ndarray,
        acoustic_features: np.ndarray,
    ) -> float:
        """
        Compute correlation between mouth movements and audio.

        Args:
            mouth_movements: Mouth movement features
            acoustic_features: Acoustic features

        Returns:
            LSE-C score (lower is better, <1.8 is good)
        """
        # Ensure same length
        min_len = min(len(mouth_movements), len(acoustic_features))
        mouth_movements = mouth_movements[:min_len]
        acoustic_features = acoustic_features[:min_len]

        # Normalize
        mouth_norm = (mouth_movements - np.mean(mouth_movements)) / (np.std(mouth_movements) + 1e-10)
        audio_norm = (acoustic_features - np.mean(acoustic_features)) / (np.std(acoustic_features) + 1e-10)

        # Compute cross-correlation
        correlation = np.abs(np.corrcoef(mouth_norm, audio_norm)[0, 1])

        # Convert correlation to LSE-C metric
        # High correlation (>0.7) -> low LSE-C (<1.8)
        lse_c = 2.0 * (1.0 - correlation)  # Scale to 0-2 range

        return float(lse_c)

    def _compute_lse_confidence(
        self,
        mouth_movements: np.ndarray,
        acoustic_features: np.ndarray,
    ) -> float:
        """Compute confidence in LSE-C measurement."""
        # Higher confidence if both signals are clear
        mouth_clarity = np.std(mouth_movements)
        audio_clarity = np.std(acoustic_features)

        confidence = min(mouth_clarity, audio_clarity)
        return float(np.clip(confidence, 0.0, 1.0))

    def _compute_alignment_offset(
        self,
        mouth_movements: np.ndarray,
        acoustic_features: np.ndarray,
    ) -> Tuple[float, int]:
        """
        Compute temporal offset between mouth and audio.

        Args:
            mouth_movements: Mouth movement features
            acoustic_features: Acoustic features

        Returns:
            Tuple of (LSE-D score, offset in frames)
        """
        # Ensure same length
        min_len = min(len(mouth_movements), len(acoustic_features))
        mouth_movements = mouth_movements[:min_len]
        acoustic_features = acoustic_features[:min_len]

        # Normalize
        mouth_norm = (mouth_movements - np.mean(mouth_movements)) / (np.std(mouth_movements) + 1e-10)
        audio_norm = (acoustic_features - np.mean(acoustic_features)) / (np.std(acoustic_features) + 1e-10)

        # Find best alignment (max cross-correlation)
        cross_corr = np.correlate(mouth_norm, audio_norm, mode="full")
        offset = np.argmax(cross_corr) - len(audio_norm) + 1

        # LSE-D score (offset in frames, lower is better)
        lse_d = min(2.0, abs(offset) * 0.1)  # Scale offset to LSE-D

        return float(lse_d), int(offset)


    def compute_from_manifest(self, manifest_path: str) -> Dict:
        """Evaluate lip-sync metrics on a JSONL manifest of video/audio pairs.

        Each line must have "video_path" and "audio_path".

        Args:
            manifest_path: Path to JSONL manifest file.

        Returns:
            Dictionary with aggregated LSE-C and LSE-D statistics.
        """
        import json as _json

        manifest = Path(manifest_path)
        if not manifest.exists():
            return {"error": f"Manifest not found: {manifest_path}"}

        lse_c_scores = []
        lse_d_scores = []
        errors = 0

        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = _json.loads(line)
                video_path = sample.get("video_path", "")
                audio_path = sample.get("audio_path", "")
                if not video_path or not audio_path:
                    errors += 1
                    continue
                if not Path(video_path).exists() or not Path(audio_path).exists():
                    errors += 1
                    continue

                try:
                    c_result = self.compute_lse_c_metric(video_path, audio_path)
                    d_result = self.compute_lse_d_metric(video_path, audio_path)
                    if "lse_c" in c_result:
                        lse_c_scores.append(c_result["lse_c"])
                    if "lse_d" in d_result:
                        lse_d_scores.append(d_result["lse_d"])
                except Exception as e:
                    logger.debug(f"Lip-sync metrics failed for {video_path}: {e}")
                    errors += 1

        result = {"n_samples": len(lse_c_scores), "errors": errors}
        if lse_c_scores:
            result["lse_c"] = float(np.mean(lse_c_scores))
            result["lse_c_std"] = float(np.std(lse_c_scores))
        if lse_d_scores:
            result["lse_d"] = float(np.mean(lse_d_scores))
            result["lse_d_std"] = float(np.std(lse_d_scores))

        return result


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python lipsync_metrics.py <video.mp4> <audio.wav>")
        sys.exit(1)

    video_path = sys.argv[1]
    audio_path = sys.argv[2]

    computer = LipSyncMetricsComputer()

    lse_c_result = computer.compute_lse_c_metric(video_path, audio_path)
    lse_d_result = computer.compute_lse_d_metric(video_path, audio_path)

    print("\n=== Lip-Sync Metrics ===")
    print(f"LSE-C: {lse_c_result.get('lse_c', lse_c_result.get('error', 'N/A'))}")
    print(f"LSE-D: {lse_d_result.get('lse_d', lse_d_result.get('error', 'N/A'))}")
