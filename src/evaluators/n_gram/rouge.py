from rouge_score import rouge_scorer

from src.base import BaseEvaluator
from src.registry_decorator import register_evaluator

SUPPORTED_ROUGE_TYPE = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

class RougeEvaluator(BaseEvaluator):
    def __init__(self, rouge_type: str, **kwargs):
        super().__init__(**kwargs)
        if rouge_type not in SUPPORTED_ROUGE_TYPE:
            raise ValueError(f"Invalid ROUGE type. Supported types are: {SUPPORTED_ROUGE_TYPE}")
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)

    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):
        scores = self.scorer.score(ground_truth, candidate)
        score = scores[self.rouge_type].fmeasure
        comment = f"{self.rouge_type} n-gram overlap score between texts"
        return {"score": score, "comment": comment}

    @property
    def method_name(self):
        return f"rouge_{self.rouge_type}"


for rouge_type in SUPPORTED_ROUGE_TYPE:
    def factory(rt=rouge_type, **kwargs):
        return RougeEvaluator(rt, **kwargs)
    register_evaluator(category="ngram", name=rouge_type)(factory)