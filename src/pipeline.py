import asyncio
from config.loader import ConfigLoader
from evaluators.aggregator import Aggregator
from evaluators.registry import build_registry
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


async def run_evaluator(name, evaluator, generated, reference, timeout=30):
    try:
        result = await asyncio.wait_for(
            evaluator.evaluate(generated, reference),
            timeout=timeout
        )
        return name, result
    except Exception:
        logger.exception("%s failed", name)
        return name, None


async def run_pipeline(generated, reference):
    loader = ConfigLoader("evaluation.yaml", "runtime.yaml")
    registry = build_registry(loader.runtime_config, loader)

    output = {}

    # No global validation here; unknown methods will be logged per-layer and skipped.

    for layer, cfg in loader.eval_config.get("evaluators", {}).items():
        methods = cfg.get("methods")
        if not methods:
            logger.warning("Layer '%s' has no methods", layer)
            continue

        tasks = []
        for method in methods:
            if method not in registry:
                logger.warning(
                    "Method '%s' not registered; add it in src/evaluators/registry.py as a zero-arg factory or add a prompt in src/evaluation_prompts.yaml",
                    method,
                )
                continue

            evaluator = registry[method]()
            tasks.append(run_evaluator(method, evaluator, generated, reference))

        # 🔥 Parallel execution per layer
        results = await asyncio.gather(*tasks)

        # `results` is a list of (method_name, result_dict)
        evaluator_outputs = {k: v for k, v in results if v is not None}

        # Build numeric score mapping for aggregation
        numeric_scores = {k: v.get("score", 0.0) for k, v in evaluator_outputs.items()}

        output[layer] = {
            "final_score": Aggregator.aggregate(numeric_scores, methods),
            "evaluators": evaluator_outputs,
            "weights": methods,
        }

    return output


if __name__ == "__main__":
    result = asyncio.run(
        run_pipeline("Generated text", "Reference text")
    )
    logger.info("Pipeline result: %s", result)
