import os
import yaml

def load_runtime_config(path: str) -> dict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    
    def resolve_env(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.getenv(value[2:-1])
        return value

    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return resolve_env(obj)

    return walk(raw)
        
