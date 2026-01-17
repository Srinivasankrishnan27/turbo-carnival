import json
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import httpx

class MetaEvaluationOutput(BaseModel):
    summary: str = Field(description="A qualitative summary of the evaluation results across all metrics.")
    verdict: str = Field(description="A final verdict: 'PASS', 'FAIL', or 'NEEDS_REVIEW'.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")

from openai import AsyncOpenAI

import os
import yaml

class MetaEvaluator:
    def __init__(self, base_url, api_key, model_name, **kwargs):
        # Load prompts
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_path = os.path.join(base_dir, "prompts.yaml")
        with open(prompts_path, "r") as f:
            self.prompts = yaml.safe_load(f).get("meta_evaluator", {})

        self.agent = Agent(
            model_name,
            output_type=MetaEvaluationOutput,
            system_prompt=self.prompts.get("system_prompt", "")
        )

    async def synthesize(self, pipeline_results: dict):
        results_str = json.dumps(pipeline_results, indent=2)
        template = self.prompts.get("user_prompt_template", "")
        user_prompt = template.format(results_str=results_str)
        
        # Run agent
        result = await self.agent.run(user_prompt)
        return result.output
