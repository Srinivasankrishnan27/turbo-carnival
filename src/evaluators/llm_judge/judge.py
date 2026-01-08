from src.base import BaseEvaluator
from src.registry_decorator import register_evaluator
from src.evaluators.llm_judge.data_model import LLMJudgeEvaluatorOuput
from src.utils.timeit import timeit
import json
import httpx
import yaml
import os
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

prompt_path = os.path.join(os.path.dirname(__file__), "prompt.yaml")
with open(prompt_path, "r") as f:
    PROMPT_CFG = yaml.safe_load(f)

SYSTEM_PROMPT = PROMPT_CFG["system_prompt"]
DIMENSIONS = list(PROMPT_CFG["dimensions"].keys())

class LLMJudgeDimensionEvaluator(BaseEvaluator):
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    def __init__(self, dimension: str, base_url, api_key, model_name, stream, **kwargs):
        super().__init__(**kwargs)
        if dimension not in DIMENSIONS:
            raise ValueError(f"Invalid dimension. Supported dimensions are: {DIMENSIONS}")
        self.dimension = dimension
        self.prompt = PROMPT_CFG["dimensions"][dimension]["prompt"]
        self.few_shot = PROMPT_CFG["dimensions"][dimension].get("few_shot_examples", [])

        self.__base_url = base_url
        self.__api_key = api_key
        self.__model_name = model_name
        self.__stream = stream  
        self.__httpx_client = httpx.AsyncClient(verify=False)
        self.__client = AsyncOpenAI(
            base_url=self.__base_url, 
            api_key=self.__api_key, 
            http_client=self.__httpx_client
            )
        self._response_format = LLMJudgeEvaluatorOuput
    @property
    def method_name(self):
        return f"llm_judge_{self.dimension}"
    

    @retry(
        stop=stop_after_attempt(MAX_RETRIES), 
        wait=wait_exponential(multiplier=RETRY_DELAY, min=1, max=10),
        retry=retry_if_exception_type(RateLimitError)
    )
    async def ask_model_openai(self, messages, model_params= None,  raise_truncation_exception=False, **kwargs):
        completion = await self.__client.chat.completions.parse(
            model=self.__model_name,
            messages=messages,
            #stream_options=self.__stream,
            response_format=LLMJudgeEvaluatorOuput)
        if self.__stream:
            stream_text, has_stop = await self.process_stream(completion)
            if raise_truncation_exception and not has_stop:
                print(f"[WARN] Response is truncated")
            return stream_text
        else:

            return completion.choices[0].message.content


    async def process_stream(self, stream):
        response_text = ""
        has_stop = False
        async for chunk in stream:
            if chunk.choices[0].finish_reason == "stop":
                has_stop = True
            temp_text = chunk.choices[0].delta.content
            if temp_text is not None:
                response_text += temp_text
        return response_text, has_stop
    
    async def call_llm(self, prompt, **kwargs):
        json_schema = LLMJudgeEvaluatorOuput.model_json_schema()
        model_params = {"response_json_schema": json_schema}
        response = await self.ask_model_openai(messages=prompt, model_params=model_params, **kwargs)
        
        try:
            raw_dict = json.loads(response)
        except json.JSONDecodeError as e:
            comment = "LLM Judge output failed schema validation: return invalid JSON"
            score = 0
            return {"score": score, "comment": f"{comment}_{str(e)}"}
        try:
            raw = LLMJudgeEvaluatorOuput(**raw_dict)
        except Exception as e:
            comment = "LLM Judge output failed schema validation"
            score = 0
            return {"score": score, "comment": f"{comment}_{str(e)}"}
        response = raw.model_dump()
        return response
    
    def build_prompt(self, ground_truth, candidate):
        user_prompt = self.prompt.replace("{ground_truth}", ground_truth).replace("{generated}", candidate)
        if self.few_shot:
            examples_text = "\n".join(
                f"Reference: {ex['reference']}\nGenerated: {ex['generated']}\nScore: {ex['score']}\nComment: {ex['comment']}\n"
                for ex in self.few_shot
            )
            user_prompt = examples_text + "\n" + user_prompt
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        return prompt
    

    @timeit
    async def evaluate(self, ground_truth: str, candidate: str, **kwargs):        
        full_prompt = self.build_prompt(ground_truth, candidate)
        result = await self.call_llm(full_prompt, **kwargs)
        return result



for dim in DIMENSIONS:
    def factory(dimension_name=dim, **kwargs):
        return LLMJudgeDimensionEvaluator(dimension_name, **kwargs)
    register_evaluator(category="llm_judge", name=dim)(factory)