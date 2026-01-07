import asyncio
import traceback

import yaml

from src.aggregator import Aggregator
from src.registry import REGISTRY
from src.runtime_config import load_runtime_config
import src.evaluators

async def run_pipeline(ground_truth: str, candidate: str, config_path: str, runtime_config_path: str):
    with open(config_path, "r") as f:
        eval_config = yaml.safe_load(f)
    
    runtime_cfg = load_runtime_config(runtime_config_path)

    results = {}

    async def evaluate_method(category: str, method: str, weight: float):
        
        kwargs = runtime_cfg.get(method, {})
        evaluator = REGISTRY.get(category, method, **kwargs)
        
        if evaluator is None:
            print(f"[WARN] Evaluator '{category}'.'{method}' not registered - Skipping")
            return method, None, weight
        
        try:
            result = await evaluator.evaluate(ground_truth, candidate)
            return method, result, weight
        except Exception as e:
            print(f"[ERROR] Failed to evaluate '{category}'.'{method}': Error: {str(e)} - Skipping")
            return method, None, weight
    
    layer_tasks = []

    for category, methods in eval_config.get("evaluators", {}).items():
        if not isinstance(methods, dict):
            print(f"[WARN] methods for {category} is not a dict - Skipping")
            continue
        tasks = []
        for method, weight in methods.items():
            weight = weight if weight is not None else 1.0
            tasks.append(evaluate_method(category, method, weight))
        layer_tasks.append((category, tasks))

    
    # Run all the layers concurrently 

    all_layer_results = await asyncio.gather(*[asyncio.gather(*tasks, return_exceptions=False) for _, tasks in layer_tasks])

    # collect results

    for (category, _), layer_results in zip(layer_tasks, all_layer_results):
        scores = {}
        weights = {}
        for method, result, weight in layer_results:
            if result is None:
                continue
            scores[method] = result
            weights[method] = weight

            if not scores:
                print(f"[INFO] No valid results for the category: {category}")
                continue
            final_score = Aggregator.aggregate({k: v["score"] for k, v in scores.items()}, weights)
            
            results[category] = {
                "final_score": final_score, 
                "evaluators": scores, 
                "weights" : weights
            }
    return results    