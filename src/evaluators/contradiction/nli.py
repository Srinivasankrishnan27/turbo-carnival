from transformers import pipeline
from src.base import BaseEvaluator
from src.utils.timeit import timeit


class NLIEvaluator(BaseEvaluator):
    def __init__(self, model: str):
        self.pipe = pipeline("text-classification", model=model)

    @timeit
    async def evaluate(self, generated, reference):
        result = self.pipe(
            f"{reference} </s></s> {generated}",
            return_all_scores=True
        )[0]

        contradiction = next(r for r in result if r["label"] == "CONTRADICTION")
        return round(1 - contradiction["score"], 4)
