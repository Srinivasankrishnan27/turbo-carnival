from evaluators.base import BaseEvaluator
from src.utils.timeit import timeit
import httpx
from openai import AsyncOpenAI, RateLimitError
from litellm import aembedding
import math
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class EmbeddingEvaluator(BaseEvaluator):
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    def __init__(self, embedding_url: str, api_key: str, embedding_model_name: str, provider: str="openai", **kwargs):
        super().__init__(**kwargs)
        self.__base_url = embedding_url
        self.__api_key = api_key
        self.__embedding_model_name = embedding_model_name
        self.__async_client = httpx.AsyncClient(verify=False)
        self.__openai_client = AsyncOpenAI(
            base_url=self.__base_url, 
            api_key=self.__api_key, 
            http_client=self.__async_client
            )
        self.__provider = provider
        self.__model_name = f"{self.__provider}/{self.__embedding_model_name}"

    @retry(retry=retry_if_exception_type(RateLimitError),
           wait=wait_exponential(multiplier=RETRY_DELAY, min=RETRY_DELAY, max=30),
           stop=stop_after_attempt(MAX_RETRIES),
           reraise=True)
    async def __get_embeddings(self, doc):
        response = await aembedding(model=self.__model_name, input=doc, custom_llm_provider=self.__provider, client=self.__openai_client)
        # safe access
        if not getattr(response, "data", None):
            raise RuntimeError("Empty embedding response")
        emb = response.data[0].get("embedding")
        if emb is None:
            raise RuntimeError("Embedding missing in response")
        return emb

    async def __get_magnitude(self, vec):
        return math.sqrt(sum(x**2 for x in vec))
    
    async def __get_dot_product(self, vec1, vec2):
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must be of equal length")
        return sum(x*y for x, y in zip(vec1, vec2))

    @timeit
    async def evaluate(self, ground_truth, candidate):
        score = 0
        ground_truth =  ground_truth.lower()
        candidate =  candidate.lower()
        
        if ground_truth == candidate:
            score = 1
        try:
            vector_gt = await self.__get_embeddings(ground_truth)
            vector_cdt = await self.__get_embeddings(candidate)

            magnitude_gt =  await self.__get_magnitude(vector_gt)
            magnitude_cdt =  await self.__get_magnitude(vector_cdt)

            dot_product = await self.__get_dot_product(vector_gt, vector_cdt)

            if magnitude_gt == 0 or magnitude_cdt == 0:
                score = 0
            else:
                score = dot_product/(magnitude_cdt * magnitude_gt)
                score = max(0.0, min(score, 1.0))
        except Exception as e:
            score = 0
        finally:
            explanation = f"Cosine similarity between embeddings (model={self.__model_name})"
            return {"method": self.method_name, "score": score, "explanation": explanation}
    
    @property
    def method_name(self):
        return "embedding"