"""
Token F1 Evaluator - Deterministic token-level precision/recall/F1 scoring.

This evaluator computes the F1 score based on token (word) overlap between
ground truth and candidate texts. It's fully deterministic and requires no
external API calls.
"""

import asyncio
from typing import Set

from src.base import BaseEvaluator
from src.registry_decorator import register_evaluator
from src.utils.timeit import timeit

try:
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


@register_evaluator(category="token_level", name="token_f1")
class TokenF1Evaluator(BaseEvaluator):
    """
    Computes Token F1 score between ground truth and candidate texts.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Where:
    - Precision = |common_tokens| / |candidate_tokens|
    - Recall = |common_tokens| / |ground_truth_tokens|
    """

    def __init__(
        self,
        lowercase: bool = True,
        use_stemming: bool = False,
        remove_stopwords: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lowercase = lowercase
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords

        # Initialize stemmer if needed
        self._stemmer = None
        if self.use_stemming and NLTK_AVAILABLE:
            self._stemmer = PorterStemmer()

        # Load stopwords if needed
        self._stopwords = set()
        if self.remove_stopwords and NLTK_AVAILABLE:
            try:
                self._stopwords = set(stopwords.words('english'))
            except LookupError:
                import nltk
                nltk.download('stopwords', quiet=True)
                self._stopwords = set(stopwords.words('english'))

    @property
    def method_name(self):
        return "token_f1"

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into a set of tokens with optional processing."""
        # Apply lowercase if configured
        if self.lowercase:
            text = text.lower()

        # Tokenize
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except LookupError:
                import nltk
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                tokens = word_tokenize(text)
        else:
            # Fallback to simple split
            tokens = text.split()

        # Remove stopwords if configured
        if self.remove_stopwords and self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        # Apply stemming if configured
        if self.use_stemming and self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        # Filter out punctuation-only tokens
        tokens = [t for t in tokens if any(c.isalnum() for c in t)]

        return set(tokens)

    def _compute_f1(self, gt_tokens: Set[str], cand_tokens: Set[str]) -> dict:
        """Compute precision, recall, and F1 score."""
        if not gt_tokens and not cand_tokens:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if not gt_tokens:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

        if not cand_tokens:
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

        common = gt_tokens & cand_tokens

        precision = len(common) / len(cand_tokens)
        recall = len(common) / len(gt_tokens)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    @timeit
    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):
        """
        Evaluate token F1 between ground truth and candidate.

        Returns:
            dict with 'score' (F1) and 'comment' with details
        """
        try:
            # Run tokenization in thread pool to avoid blocking
            gt_tokens = await asyncio.to_thread(self._tokenize, ground_truth)
            cand_tokens = await asyncio.to_thread(self._tokenize, candidate)

            # Compute F1
            metrics = self._compute_f1(gt_tokens, cand_tokens)

            comment = (
                f"Token F1: {metrics['f1']:.4f} "
                f"(P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, "
                f"GT tokens: {len(gt_tokens)}, Cand tokens: {len(cand_tokens)})"
            )

            return {"score": metrics['f1'], "comment": comment}

        except Exception as e:
            return {"score": 0.0, "comment": f"Token F1 evaluation failed: {str(e)}"}
