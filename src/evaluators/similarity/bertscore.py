"""
BERTScore Evaluator - Token-level semantic similarity using BERT embeddings.

BERTScore computes similarity between two texts by:
1. Getting contextualized token embeddings from BERT
2. Computing token-level cosine similarity matrix
3. Greedily matching tokens to compute precision and recall
4. Returning F1 score

This provides a more nuanced semantic similarity than embedding-based approaches
as it operates at the token level rather than sentence level.
"""

import asyncio
from typing import Optional

from src.base import BaseEvaluator
from src.registry_decorator import register_evaluator
from src.utils.timeit import timeit

# Try to import bert_score
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


@register_evaluator(category="similarity", name="bertscore")
class BERTScoreEvaluator(BaseEvaluator):
    """
    Computes BERTScore between ground truth and candidate texts.

    BERTScore is a deterministic metric that measures semantic similarity
    at the token level using pre-trained BERT embeddings.

    Score ranges:
    - 0.9-1.0: Nearly identical texts
    - 0.7-0.9: Very similar semantically
    - 0.5-0.7: Moderate similarity
    - 0.3-0.5: Low similarity
    - 0.0-0.3: Unrelated
    """

    def __init__(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        device: str = "cpu",
        num_layers: Optional[int] = None,
        batch_size: int = 64,
        lang: str = "en",
        rescale_with_baseline: bool = False,
        **kwargs
    ):
        """
        Initialize the BERTScore evaluator.

        Args:
            model_type: Pre-trained model to use for embeddings
            device: "cpu" or "cuda" for GPU acceleration
            num_layers: Number of layers to use (None = use all)
            batch_size: Batch size for processing
            lang: Language code for multilingual models
            rescale_with_baseline: Whether to rescale scores with baseline
        """
        super().__init__(**kwargs)
        self.model_type = model_type
        self.device = device
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline

    @property
    def method_name(self):
        return "bertscore"

    def _compute_bertscore(
        self,
        ground_truth: str,
        candidate: str
    ) -> dict:
        """Compute BERTScore metrics."""
        if not BERTSCORE_AVAILABLE:
            raise ImportError(
                "bert-score is not installed. Install with: pip install bert-score"
            )

        # BERTScore expects lists
        P, R, F1 = bert_score_fn(
            cands=[candidate],
            refs=[ground_truth],
            model_type=self.model_type,
            device=self.device,
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            lang=self.lang,
            rescale_with_baseline=self.rescale_with_baseline,
            verbose=False
        )

        return {
            "precision": P[0].item(),
            "recall": R[0].item(),
            "f1": F1[0].item()
        }

    @timeit
    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):
        """
        Evaluate BERTScore between ground truth and candidate.

        Returns:
            dict with 'score' (F1) and 'comment' with details
        """
        try:
            if not BERTSCORE_AVAILABLE:
                return {
                    "score": 0.0,
                    "comment": "bert-score not installed. Run: pip install bert-score"
                }

            # Run BERTScore computation in thread pool (CPU/GPU bound)
            metrics = await asyncio.to_thread(
                self._compute_bertscore,
                ground_truth,
                candidate
            )

            # Interpret the score
            f1 = metrics['f1']
            if f1 >= 0.9:
                interpretation = "Nearly identical"
            elif f1 >= 0.7:
                interpretation = "Very similar"
            elif f1 >= 0.5:
                interpretation = "Moderately similar"
            elif f1 >= 0.3:
                interpretation = "Low similarity"
            else:
                interpretation = "Unrelated"

            comment = (
                f"BERTScore F1: {f1:.4f} ({interpretation}) "
                f"[P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}]"
            )

            return {"score": f1, "comment": comment}

        except ImportError as e:
            return {"score": 0.0, "comment": str(e)}
        except Exception as e:
            return {
                "score": 0.0,
                "comment": f"BERTScore evaluation failed: {str(e)}"
            }
