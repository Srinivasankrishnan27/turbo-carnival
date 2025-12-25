import yaml
from typing import Dict, Any


class PromptLoader:
    """Loads prompt configurations from a YAML file.

    The YAML should contain a top-level `judges` mapping where each judge
    has keys like `system_persona`, `user_prompt_template`, `model_params`,
    and optional `response_schema` metadata.
    """

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

    def get_judges(self) -> Dict[str, Dict[str, Any]]:
        return {j["name"]: j for j in (self.cfg.get("judges") or [])}

    def get(self, name: str) -> Dict[str, Any]:
        return self.get_judges().get(name, {})
