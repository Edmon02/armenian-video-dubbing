#!/usr/bin/env python3
"""
Comprehensive evaluation suite for Armenian Video Dubbing AI.

Computes all quality metrics:
  - WER / CER (ASR)
  - MOS estimation (TTS)
  - Speaker similarity (voice cloning)
  - COMET (translation)
  - LSE-C/D (lip-sync)

Usage:
    python scripts/training/evaluate_all_models.py --asr-model models/asr/whisper-hy-full --output-dir outputs/evaluation
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.training_utils import MetricsComputer, load_jsonl_manifest


class ComprehensiveEvaluator:
    """Run full evaluation suite."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    @staticmethod
    def _not_measured(stage: str, how: str) -> dict:
        """Placeholder result: this suite has no real scorer for the stage yet."""
        logger.warning("{}: not measured. {}", stage, how)
        return {"status": "not_measured", "note": how}

    def evaluate_asr(self, test_manifest: Path, model_path: Path) -> dict:
        """Evaluate ASR model on test set."""
        logger.info("=" * 60)
        logger.info("ASR Evaluation (WER / CER)")
        logger.info("=" * 60)

        try:
            test_data = load_jsonl_manifest(test_manifest)
        except Exception as e:
            logger.error("Failed to load test data: {}", e)
            return {}

        logger.info("Test set: {} samples", len(test_data))
        result = self._not_measured(
            "ASR", "Run scripts/evaluation/metrics/wer_metrics.py or the notebook ASR table."
        )
        result.update({"dataset": str(test_manifest), "num_samples": len(test_data)})
        return result

    def evaluate_tts(self, reference_samples: list[tuple[str, str]]) -> dict:
        """Evaluate TTS on MOS + speaker similarity."""
        logger.info("=" * 60)
        logger.info("TTS Evaluation (MOS / Speaker Similarity)")
        logger.info("=" * 60)
        return self._not_measured(
            "TTS", "Use UTMOSv2, SECS and back-ASR CER from notebooks/colab_dubbing_ablation.ipynb."
        )

    def evaluate_translation(self) -> dict:
        """Evaluate translation."""
        logger.info("=" * 60)
        logger.info("Translation Evaluation (COMET)")
        logger.info("=" * 60)
        return self._not_measured(
            "Translation",
            "Use TranslationQualityComputer.compute_comet_batch with reference translations.",
        )

    def evaluate_lipsync(self) -> dict:
        """Evaluate lip-sync (LSE-C/D metrics)."""
        logger.info("=" * 60)
        logger.info("Lip-Sync Evaluation (LSE-C/D)")
        logger.info("=" * 60)
        return self._not_measured(
            "Lip-sync", "Use the SyncNet scorer in notebooks/colab_dubbing_ablation.ipynb."
        )

    def run_full_evaluation(self, test_manifest: Path, asr_model: Path) -> dict:
        """Run all evaluations."""
        self.results = {
            "timestamp": str(Path.ctime(Path.cwd())),
            "models": {
                "asr": str(asr_model),
            },
            "metrics": {},
        }

        # ASR
        self.results["metrics"]["asr"] = self.evaluate_asr(test_manifest, asr_model)

        # TTS (placeholder reference samples)
        self.results["metrics"]["tts"] = self.evaluate_tts(
            [("Բարեւ", "Hello"), ("Շատ լավ", "Very good")]
        )

        # Translation
        self.results["metrics"]["translation"] = self.evaluate_translation()

        # Lip-sync
        self.results["metrics"]["lipsync"] = self.evaluate_lipsync()

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Evaluation Summary")
        logger.info("=" * 60)

        # Check targets (None = not measured)
        m = self.results["metrics"]

        def check(value, ok):
            return None if value is None else ok(value)

        targets_met = {
            "WER <8%": check(m["asr"].get("wer"), lambda v: v < 0.08),
            "MOS >4.6": check(m["tts"].get("mos_mean"), lambda v: v > 4.6),
            "Speaker Similarity >0.85": check(m["tts"].get("speaker_similarity_mean"), lambda v: v > 0.85),
            "LSE-C <1.8": check(m["lipsync"].get("lse_c"), lambda v: v < 1.8),
            "LSE-D <1.8": check(m["lipsync"].get("lse_d"), lambda v: v < 1.8),
        }

        for target, met in targets_met.items():
            status = "?" if met is None else ("✓" if met else "✗")
            logger.info("  {} {}", status, target)

        measured = [v for v in targets_met.values() if v is not None]
        logger.info("")
        logger.info("Targets met: {}/{} measured ({} not measured)",
                    sum(measured), len(measured), len(targets_met) - len(measured))

        return self.results

    def save_results(self):
        """Save evaluation results."""
        output_file = self.output_dir / "evaluation_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Saved evaluation results to {}", output_file)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument("--test-manifest", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--asr-model", type=str, default="models/asr/whisper-hy-full")
    parser.add_argument("--tts-model", type=str, default="models/tts/fish-speech-hy")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")

    args = parser.parse_args()
    setup_logger()

    evaluator = ComprehensiveEvaluator(Path(args.output_dir))

    results = evaluator.run_full_evaluation(
        test_manifest=Path(args.test_manifest),
        asr_model=Path(args.asr_model),
    )

    evaluator.save_results()

    logger.info("")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
