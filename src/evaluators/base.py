from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):
            """Evaluate a (generated, reference) pair.

            Must return a mapping with at least:
                - "method": str
                - "score": float
            Optionally include:
                - "explanation": str  # human-readable justification
            Example: {"method": "bertscore", "score": 0.82, "explanation": "F1 between texts"}
            """
            pass
