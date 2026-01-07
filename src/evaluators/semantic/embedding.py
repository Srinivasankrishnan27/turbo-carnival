import logging
import httpx
from openai import AsyncOpenAI, RateLimitError
from litellm import aembedding
import math
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.base import BaseEvaluator
from src.registry_decorator import register_evaluator
import traceback

@register_evaluator(category="semantic", name="embedding")
class EmbeddingSimilarityEvaluator(BaseEvaluator):
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    def __init__(self, base_url: str, api_key: str, embedding_model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.__base_url = base_url
        self.__api_key = api_key
        self.__embedding_model_name = embedding_model_name
        self.__async_client = httpx.AsyncClient(verify=False)
        self.__openai_client = AsyncOpenAI(
            base_url=self.__base_url, 
            api_key=self.__api_key, 
            http_client=self.__async_client
            )    
        self.__model_name = {self.__embedding_model_name}
    
    @property
    def method_name(self):
        return "embedding"

    @retry(
        stop=stop_after_attempt(MAX_RETRIES), 
        wait=wait_exponential(multiplier=RETRY_DELAY, min=1, max=10),
        retry=retry_if_exception_type(RateLimitError)
    )
    async def get_embeddings(self, doc):
        response = await self.__openai_client.embeddings.create(
            input=doc,
            model=self.__embedding_model_name
        )
        return response.data[0].embedding
    
    async def magnitude(self, vec):
        return math.sqrt(sum(x**2 for x in vec))
    
    async def dot_product(self, vec1, vec2):
        return sum(x * y for x, y in zip(vec1, vec2))
    
    async def evaluate(self, ground_truth, candidate, **kwargs):
        score = 0
        try:
            if ground_truth.lower() == candidate.lower():
                score = 1
            else:
                gt_embedding = await self.get_embeddings(ground_truth.lower())
                candidate_embedding = await self.get_embeddings(candidate.lower())
                magnitude_gt = await self.magnitude(gt_embedding)
                magnitude_candidate = await self.magnitude(candidate_embedding)
                if magnitude_candidate == 0 or magnitude_gt == 0:
                    score = 0
                else:
                    dot_product = await self.dot_product(gt_embedding, candidate_embedding)
                    score = dot_product / (magnitude_gt * magnitude_candidate)
        except Exception as e:
            logging.error(traceback.format_exc())
            score = 0
        finally:
            logging.info(f"cosine similarity is {score}")
            comment = "High semantic similarity" if score > 0.8 else "Moderate semantic similarity"
            return {"score": score, "comment": comment}