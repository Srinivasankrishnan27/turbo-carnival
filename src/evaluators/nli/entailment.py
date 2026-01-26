"""
NLI (Natural Language Inference) Evaluators - Entailment and Contradiction detection.

These evaluators use pre-trained NLI models to determine the logical relationship
between ground truth (premise) and candidate (hypothesis):

- Entailment: The candidate logically follows from the ground truth
- Neutral: The candidate is neither supported nor contradicted
- Contradiction: The candidate contradicts the ground truth

This provides strong deterministic evidence for:
- Factual accuracy (high entailment = facts are supported)
- Hallucination detection (high contradiction = fabricated/wrong information)
"""

import asyncio
from typing import Dict, Optional

from src.base import BaseEvaluator
from src.registry_decorator import register_evaluator
from src.utils.timeit import timeit

# Try to import transformers
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BaseNLIEvaluator(BaseEvaluator):
    """
    Base class for NLI-based evaluators.

    Loads and caches the NLI model for inference.
    """

    # Class-level model cache to share across instances
    _model_cache: Dict[str, tuple] = {}

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: str = "cpu",
        max_length: int = 512,
        **kwargs
    ):
        """
        Initialize the NLI evaluator.

        Args:
            model_name: Pre-trained NLI model from Hugging Face
            device: "cpu" or "cuda" for GPU acceleration
            max_length: Maximum sequence length for tokenization
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        # Model will be lazy-loaded
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load or retrieve cached model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are not installed. "
                "Install with: pip install transformers torch"
            )

        # Check class-level cache first
        if self.model_name in self._model_cache:
            self._tokenizer, self._model = self._model_cache[self.model_name]
            return

        # Load model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model.to(self.device)
        self._model.eval()

        # Cache for future use
        self._model_cache[self.model_name] = (self._tokenizer, self._model)

    def _run_inference(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Run NLI inference.

        Args:
            premise: The ground truth text
            hypothesis: The candidate text to evaluate

        Returns:
            Dict with probabilities for entailment, neutral, contradiction
        """
        if self._model is None:
            self._load_model()

        # Tokenize
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        # Map indices to labels
        # Most NLI models use: 0=contradiction, 1=neutral, 2=entailment
        # But some use different orderings, so we check the config
        id2label = self._model.config.id2label

        result = {}
        for idx, prob in enumerate(probs):
            label = id2label.get(idx, str(idx)).lower()
            result[label] = prob.item()

        return result


@register_evaluator(category="nli", name="entailment")
class EntailmentEvaluator(BaseNLIEvaluator):
    """
    Evaluates whether the candidate text is entailed by the ground truth.

    High entailment score means the candidate logically follows from
    the ground truth - indicating factual consistency.

    Score interpretation:
    - 0.9-1.0: Strong entailment - candidate is strongly supported
    - 0.7-0.9: Moderate entailment - candidate is mostly supported
    - 0.5-0.7: Weak entailment - some support
    - 0.0-0.5: Low/no entailment - candidate may not be supported
    """

    @property
    def method_name(self):
        return "nli_entailment"

    @timeit
    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):
        """
        Evaluate entailment between ground truth and candidate.

        Returns:
            dict with 'score' (entailment probability) and 'comment'
        """
        try:
            if not TRANSFORMERS_AVAILABLE:
                return {
                    "score": 0.0,
                    "comment": "transformers/torch not installed"
                }

            # Run NLI inference in thread pool
            probs = await asyncio.to_thread(
                self._run_inference,
                ground_truth,
                candidate
            )

            # Get entailment probability
            entailment_score = probs.get('entailment', 0.0)
            neutral_score = probs.get('neutral', 0.0)
            contradiction_score = probs.get('contradiction', 0.0)

            # Interpret
            if entailment_score >= 0.9:
                interpretation = "Strongly entailed"
            elif entailment_score >= 0.7:
                interpretation = "Moderately entailed"
            elif entailment_score >= 0.5:
                interpretation = "Weakly entailed"
            else:
                interpretation = "Not entailed"

            comment = (
                f"Entailment: {entailment_score:.4f} ({interpretation}) "
                f"[N: {neutral_score:.4f}, C: {contradiction_score:.4f}]"
            )

            return {"score": entailment_score, "comment": comment}

        except ImportError as e:
            return {"score": 0.0, "comment": str(e)}
        except Exception as e:
            return {
                "score": 0.0,
                "comment": f"Entailment evaluation failed: {str(e)}"
            }


@register_evaluator(category="nli", name="contradiction")
class ContradictionEvaluator(BaseNLIEvaluator):
    """
    Evaluates whether the candidate text contradicts the ground truth.

    This is a HALLUCINATION DETECTOR - high score means NO contradiction
    (i.e., the candidate is factually consistent).

    The score is INVERTED: score = 1 - P(contradiction)
    So higher scores are better (less contradiction).

    Score interpretation:
    - 0.9-1.0: No contradiction detected - factually consistent
    - 0.7-0.9: Low contradiction risk
    - 0.5-0.7: Moderate contradiction risk
    - 0.0-0.5: HIGH CONTRADICTION - likely hallucination/error
    """

    @property
    def method_name(self):
        return "nli_contradiction"

    @timeit
    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):
        """
        Evaluate contradiction between ground truth and candidate.

        Returns:
            dict with 'score' (1 - contradiction probability) and 'comment'
        """
        try:
            if not TRANSFORMERS_AVAILABLE:
                return {
                    "score": 0.0,
                    "comment": "transformers/torch not installed"
                }

            # Run NLI inference in thread pool
            probs = await asyncio.to_thread(
                self._run_inference,
                ground_truth,
                candidate
            )

            # Get contradiction probability and invert it
            contradiction_prob = probs.get('contradiction', 0.0)
            entailment_score = probs.get('entailment', 0.0)
            neutral_score = probs.get('neutral', 0.0)

            # Invert so higher = better (no contradiction)
            score = 1.0 - contradiction_prob

            # Interpret based on contradiction probability
            if contradiction_prob >= 0.5:
                interpretation = "HIGH CONTRADICTION - Possible hallucination!"
            elif contradiction_prob >= 0.3:
                interpretation = "Moderate contradiction risk"
            elif contradiction_prob >= 0.1:
                interpretation = "Low contradiction risk"
            else:
                interpretation = "No contradiction detected"

            comment = (
                f"Non-contradiction: {score:.4f} ({interpretation}) "
                f"[E: {entailment_score:.4f}, N: {neutral_score:.4f}, "
                f"C: {contradiction_prob:.4f}]"
            )

            return {"score": score, "comment": comment}

        except ImportError as e:
            return {"score": 0.0, "comment": str(e)}
        except Exception as e:
            return {
                "score": 0.0,
                "comment": f"Contradiction evaluation failed: {str(e)}"
            }
