from abc import ABC, abstractmethod


class BaseEvaluator(ABC):

    def __init__(self, **kwargs):
          self.runtime_kwargs = kwargs

    @abstractmethod
    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):
            """Evaluate a (generated, reference) pair.
            Must return a mapping with at least:
            Example: {"score": 0.85, "comment": "some comments"}
            """
            raise NotImplementedError("Each evaluator must implement the evaluate method")
