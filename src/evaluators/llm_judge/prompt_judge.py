from typing import Any, Dict

from jinja2 import Template
from pydantic import BaseModel, Field, confloat

from src.evaluators.llm_judge.base_judge import BaseLLMJudge


class PromptResponseModel(BaseModel):
	score: confloat(ge=0.0, le=1.0)
	method: str
	comments: str = Field("", max_length=2000)


class PromptJudge(BaseLLMJudge):
	"""YAML-driven judge that renders a template and validates response.

	`cfg` should contain `user_prompt_template`, optional `system_persona`,
	and optional `model_params`. The prompt YAML loader should ensure `name`
	is present on the cfg.
	"""

	def __init__(self, cfg: Dict[str, Any], api_key: str):
		self._cfg = cfg
		# synthesize minimal llm_cfg for BaseLLMJudge
		llm_cfg = type("LC", (), {})()
		llm_cfg.model = cfg.get("model", "")
		llm_cfg.base_url = cfg.get("base_url", "https://api.openai.com/v1")
		llm_cfg.temperature = cfg.get("temperature", 0.0)
		llm_cfg.timeout = cfg.get("timeout", 30)
		llm_cfg.stream = cfg.get("stream", False)
		llm_cfg.model_params = cfg.get("model_params", {})

		super().__init__(llm_cfg, api_key)

	async def evaluate(self, generated: str, reference: str) -> Dict[str, Any]:
		tpl = Template(self._cfg.get("user_prompt_template", ""))
		user = tpl.render(generated=generated, reference=reference)
		system = self._cfg.get("system_persona")

		messages = []
		if system:
			messages.append({"role": "system", "content": system})
		messages.append({"role": "user", "content": user})

		resp = await self.call_llm(messages, model_params=self._cfg.get("model_params"), response_schema=PromptResponseModel)

		# If parsed into Pydantic model, return normalized dict
		if isinstance(resp, PromptResponseModel):
			return {"method": resp.method, "score": float(resp.score), "explanation": resp.comments}

		# If resp is a dict parsed by provider, try to map
		if isinstance(resp, dict):
			method = resp.get("method", self._cfg.get("name", "prompt_judge"))
			score = float(resp.get("score", 0.0))
			comments = resp.get("comments", "")
			return {"method": method, "score": score, "explanation": comments}

		# fallback: raw text
		raw = resp if isinstance(resp, str) else str(resp)
		try:
			import json

			parsed = json.loads(raw)
			method = parsed.get("method", self._cfg.get("name", "prompt_judge"))
			score = float(parsed.get("score", 0.0))
			comments = parsed.get("comments", raw)
			return {"method": method, "score": score, "explanation": comments}
		except Exception:
			import re

			m = re.search(r"([0-9]*\.?[0-9]+)", raw)
			score = float(m.group(1)) if m else 0.0
			return {"method": self._cfg.get("name", "prompt_judge"), "score": score, "explanation": raw.strip()}

