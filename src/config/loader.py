import os
import yaml
from dotenv import load_dotenv
from config.schemas import RuntimeConfig

load_dotenv()


class ConfigLoader:
    def __init__(self, eval_path: str, runtime_path: str):
        self.eval_config = self._load_yaml(eval_path)
        self.runtime_config = self._load_runtime(runtime_path)

    def _load_yaml(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _load_runtime(self, path: str) -> RuntimeConfig:
        return RuntimeConfig(**self._load_yaml(path))

    def get_api_key(self, env_name: str) -> str:
        value = os.getenv(env_name)
        if not value:
            raise RuntimeError(f"Missing env var: {env_name}")
        return value
