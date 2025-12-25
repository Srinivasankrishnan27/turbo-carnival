from rouge_score import rouge_scorer
from evaluators.base import BaseEvaluator
from src.utils.logging_config import get_logger
from src.utils.timeit import timeit


logger = get_logger(__name__)


class RougeEvaluator(BaseEvaluator):
    def __init__(self, rouge_type: str):
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)

    @timeit
    async def evaluate(self, generated, reference):
        try:
            s = self.scorer.score(reference, generated)
            score = s[self.rouge_type].fmeasure
            explanation = f"ROUGE-{self.rouge_type} F1 measure"
            return {"method": self.method_name, "score": score, "explanation": explanation}
        except Exception as e:
            logger.exception("ROUGE failed")
            return {"method": self.method_name, "score": 0.0, "explanation": str(e)}

    @property
    def method_name(self):
        return f"rouge_{self.rouge_type}"
