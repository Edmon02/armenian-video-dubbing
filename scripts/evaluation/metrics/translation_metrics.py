#!/usr/bin/env python3
"""
Translation Quality Metrics (COMET, METEOR, BERTScore).

Evaluates Armenian translation quality from source language.

Metrics:
- COMET: Neural MT evaluation (best for evaluating MT quality)
- METEOR: Handles synonyms, paraphrases
- BERTScore: Contextual embedding-based metric
- Semantic similarity: Meaning preservation
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
from loguru import logger

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    logger.warning("COMET not available (optional)")

try:
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    logger.warning("METEOR not available (optional)")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("BERTScore not available (optional)")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence-transformers not available (optional)")

from src.utils.helpers import timer


class TranslationQualityComputer:
    """Compute translation quality metrics."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize translation quality computer.

        Args:
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device

        # Load COMET model
        if COMET_AVAILABLE:
            try:
                model_path = download_model("Unbabel/wmt22-comet-da")
                self.comet_model = load_from_checkpoint(model_path)
                self.comet_model = self.comet_model.to(device)
                logger.info("Loaded COMET model")
            except Exception as e:
                logger.warning(f"Failed to load COMET: {e}")
                self.comet_model = None
        else:
            self.comet_model = None

        # Load multilingual embedder for semantic similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(
                    "sentence-transformers/distiluse-base-multilingual-cased-v2",
                    device=device
                )
                logger.info("Loaded multilingual embedder")
            except Exception as e:
                logger.warning(f"Failed to load embedder: {e}")
                self.embedder = None
        else:
            self.embedder = None

        logger.info("TranslationQualityComputer initialized")

    def compute_comet_score(
        self,
        source_text: str,
        target_text: str,
        reference_text: Optional[str] = None,
    ) -> Dict:
        """
        Compute COMET score (neural MT evaluation).

        Args:
            source_text: Source language text
            target_text: Translated text (Armenian)
            reference_text: Optional reference translation

        Returns:
            Dictionary with COMET score
        """
        result = self.compute_comet_batch(
            [source_text], [target_text], [reference_text] if reference_text else None
        )
        if "error" in result:
            return result
        return {"comet_score": result["comet_score"], "n_samples": 1}

    def compute_comet_batch(
        self,
        source_list: List[str],
        target_list: List[str],
        reference_list: Optional[List[Optional[str]]] = None,
        batch_size: int = 16,
    ) -> Dict:
        """Score translations with wmt22-comet-da in one batched call.

        wmt22-comet-da is reference-based, so samples without a reference are skipped.
        """
        if self.comet_model is None:
            logger.warning("COMET model not available")
            return {"error": "COMET not available"}

        if not reference_list:
            return {"error": "wmt22-comet-da needs reference translations"}

        samples = [
            {"src": s, "mt": t, "ref": r}
            for s, t, r in zip(source_list, target_list, reference_list)
            if r
        ]
        if not samples:
            return {"error": "wmt22-comet-da needs reference translations"}

        try:
            with timer("COMET computation"):
                gpus = 1 if str(self.device).startswith("cuda") and torch.cuda.is_available() else 0
                output = self.comet_model.predict(samples, batch_size=batch_size, gpus=gpus)
            scores = [float(s) for s in output.scores]
            return {
                "comet_score": float(np.mean(scores)),
                "scores": scores,
                "n_samples": len(scores),
            }
        except Exception as e:
            logger.error(f"COMET computation failed: {e}")
            return {"error": str(e)}

    def compute_meteor_score(
        self,
        hypothesis: str,
        reference: str,
    ) -> Dict:
        """
        Compute METEOR score (handles synonyms, paraphrases).

        Args:
            hypothesis: Translated text
            reference: Reference translation

        Returns:
            Dictionary with METEOR score
        """
        if not METEOR_AVAILABLE:
            return {"score": 0.5, "note": "METEOR not available"}

        try:
            with timer("METEOR computation"):
                # METEOR requires tokenized input
                hyp_tokens = hypothesis.split()
                ref_tokens = reference.split()

                score = meteor_score([ref_tokens], hyp_tokens)

                return {
                    "meteor_score": float(score),
                }

        except Exception as e:
            logger.error(f"METEOR computation failed: {e}")
            return {"error": str(e)}

    def compute_bertscore(
        self,
        hypothesis: str,
        reference: str,
        lang: str = "hye",  # Armenian
    ) -> Dict:
        """
        Compute BERTScore (contextual embedding-based).

        Args:
            hypothesis: Translated text
            reference: Reference translation
            lang: Language code

        Returns:
            Dictionary with BERTScore (precision, recall, F1)
        """
        if not BERTSCORE_AVAILABLE:
            return {
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "note": "BERTScore not available"
            }

        try:
            with timer("BERTScore computation"):
                P, R, F1 = bert_score(
                    [hypothesis],
                    [reference],
                    lang=lang,
                    device=self.device,
                )

                return {
                    "precision": float(P[0]),
                    "recall": float(R[0]),
                    "f1": float(F1[0]),
                }

        except Exception as e:
            logger.error(f"BERTScore computation failed: {e}")
            return {"error": str(e)}

    def semantic_similarity(
        self,
        source_text: str,
        target_text: str,
    ) -> Dict:
        """
        Measure semantic preservation using embeddings.

        Args:
            source_text: Source text
            target_text: Translated text

        Returns:
            Dictionary with semantic similarity
        """
        if self.embedder is None:
            return {
                "similarity": 0.5,
                "note": "Embedder not available"
            }

        try:
            with timer("Semantic similarity"):
                # Embed both texts
                src_embed = self.embedder.encode(source_text, convert_to_tensor=True)
                tgt_embed = self.embedder.encode(target_text, convert_to_tensor=True)

                # Compute cosine similarity
                similarity = util.pytorch_cos_sim(src_embed, tgt_embed)[0][0].item()

                return {
                    "semantic_similarity": float(similarity),
                    "preserves_meaning": bool(similarity > 0.5),
                }

        except Exception as e:
            logger.error(f"Semantic similarity computation failed: {e}")
            return {"error": str(e)}

    def batch_translation_evaluation(
        self,
        source_list: List[str],
        target_list: List[str],
        reference_list: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate multiple translations.

        Args:
            source_list: List of source texts
            target_list: List of translated texts (Armenian)
            reference_list: Optional list of reference translations

        Returns:
            Dictionary with batch evaluation metrics
        """
        if len(source_list) != len(target_list):
            raise ValueError("Source and target lists must have same length")

        logger.info(f"Evaluating {len(target_list)} translations")

        comet_scores = []
        meteor_scores = []
        bertscore_f1s = []
        semantic_sims = []

        if self.comet_model and reference_list:
            comet_result = self.compute_comet_batch(source_list, target_list, reference_list)
            comet_scores = comet_result.get("scores", [])

        for i, (src, tgt) in enumerate(zip(source_list, target_list)):
            # METEOR
            if METEOR_AVAILABLE:
                meteor_result = self.compute_meteor_score(tgt, src)  # Use source as reference for demo
                if "meteor_score" in meteor_result:
                    meteor_scores.append(meteor_result["meteor_score"])

            # BERTScore
            if BERTSCORE_AVAILABLE:
                bert_result = self.compute_bertscore(tgt, src)
                if "f1" in bert_result:
                    bertscore_f1s.append(bert_result["f1"])

            # Semantic similarity
            if self.embedder:
                sem_result = self.semantic_similarity(src, tgt)
                if "semantic_similarity" in sem_result:
                    semantic_sims.append(sem_result["semantic_similarity"])

        result = {
            "n_samples": len(target_list),
        }

        if comet_scores:
            result["mean_comet"] = float(np.mean(comet_scores))
            result["std_comet"] = float(np.std(comet_scores))

        if meteor_scores:
            result["mean_meteor"] = float(np.mean(meteor_scores))
            result["std_meteor"] = float(np.std(meteor_scores))

        if bertscore_f1s:
            result["mean_bertscore_f1"] = float(np.mean(bertscore_f1s))
            result["std_bertscore_f1"] = float(np.std(bertscore_f1s))

        if semantic_sims:
            result["mean_semantic_sim"] = float(np.mean(semantic_sims))
            result["std_semantic_sim"] = float(np.std(semantic_sims))

        logger.info(f"Translation evaluation complete: {result}")

        return result

    def detect_translation_failures(
        self,
        comet_scores: List[float],
        threshold: float = 0.75,
    ) -> Dict:
        """
        Identify low-quality translations.

        Args:
            comet_scores: List of COMET scores
            threshold: Minimum acceptable COMET score

        Returns:
            Dictionary with failure analysis
        """
        failed_indices = [i for i, score in enumerate(comet_scores) if score < threshold]

        if not failed_indices:
            return {
                "failures_detected": False,
                "failed_count": 0,
            }

        failed_scores = [comet_scores[i] for i in failed_indices]

        return {
            "failures_detected": True,
            "failed_count": len(failed_indices),
            "failure_rate": len(failed_indices) / len(comet_scores),
            "failed_indices": failed_indices,
            "mean_failed_score": float(np.mean(failed_scores)),
            "min_failed_score": float(np.min(failed_scores)),
        }


    def compute_from_manifest(self, manifest_path: str) -> Dict:
        """Evaluate translation quality on a JSONL manifest.

        Each line must have "source_text" and "target_text".
        Optionally "reference_text" for reference-based metrics.

        Args:
            manifest_path: Path to JSONL manifest file.

        Returns:
            Dictionary with aggregated translation quality statistics.
        """
        import json as _json

        manifest = Path(manifest_path)
        if not manifest.exists():
            return {"error": f"Manifest not found: {manifest_path}"}

        sources, targets, references = [], [], []

        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = _json.loads(line)
                src = sample.get("source_text", sample.get("src_text", ""))
                tgt = sample.get("target_text", sample.get("tgt_text", ""))
                ref = sample.get("reference_text", "")
                if src and tgt:
                    sources.append(src)
                    targets.append(tgt)
                    references.append(ref if ref else None)

        if not sources:
            return {"error": "No valid samples found", "n_samples": 0}

        result = self.batch_translation_evaluation(
            sources, targets, references if any(r for r in references) else None
        )

        if "mean_comet" in result:
            result["comet_score"] = result["mean_comet"]

        return result


if __name__ == "__main__":
    # Example usage
    computer = TranslationQualityComputer()

    # Example EN-HY translation
    source = "The quick brown fox jumps over the lazy dog"
    target = "Արագ շագանակագույն աղվեսը կիծ անում է ծույի շանը"

    print("\n=== Translation Quality Metrics ===")
    print(f"Source: {source}")
    print(f"Target: {target}")

    comet_result = computer.compute_comet_score(source, target)
    print(f"\nCOMET: {comet_result.get('comet_score', 'N/A')}")

    bertscore_result = computer.compute_bertscore(target, source)
    print(f"BERTScore F1: {bertscore_result.get('f1', 'N/A'):.4f}")

    sem_result = computer.semantic_similarity(source, target)
    print(f"Semantic Similarity: {sem_result.get('semantic_similarity', 'N/A'):.4f}")
