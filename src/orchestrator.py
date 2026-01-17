import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import os

class EvaluationPlan(BaseModel):
    reasoning: str = Field(description="Why this plan was chosen.")
    recommended_config: str = Field(description="A valid YAML string representing the configured 'evaluators' section.")


class Orchestrator:
    def __init__(self, base_url, api_key, model_name, **kwargs):
        # Load prompts
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_path = os.path.join(base_dir, "prompts.yaml")
        with open(prompts_path, "r") as f:
            self.prompts = yaml.safe_load(f).get("orchestrator", {})

        self.agent = Agent(
            model_name,
            output_type=EvaluationPlan,
            system_prompt=self.prompts.get("system_prompt", "")
        )

    async def plan_evaluation(self, ground_truth: str, candidate: str, base_config: dict):
        base_config_str = yaml.dump(base_config)
        template = self.prompts.get("user_prompt_template", "")
        user_prompt = template.format(
            base_config_str=base_config_str,
            ground_truth=ground_truth[:500],
            candidate=candidate[:500]
        )

        # Run agent
        result = await self.agent.run(user_prompt)
        return result.output
