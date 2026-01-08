import asyncio
import json
import os

from src.pipeline import run_pipeline

if __name__ == "__main__":
    ground_truth = (
        "The Apollo 11 mission successfully landed humans on the Moon on July 20, 1969."
    )
    candidate = "Humans landed on the Moon for the first time on July 20, 1969, during the successful Apollo 11 mission."
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    runtime_config_path = os.path.join(base_dir, "runtime_config.yaml")
    results = asyncio.run(
        run_pipeline(ground_truth, candidate, config_path, runtime_config_path)
    )
    print(json.dumps(results, indent=4))
