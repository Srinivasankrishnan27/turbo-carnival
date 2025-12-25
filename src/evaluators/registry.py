from src.evaluators.semantic.embedding import EmbeddingEvaluator
from src.evaluators.semantic.rouge import RougeEvaluator
from src.evaluators.contradiction.nli import NLIEvaluator
from src.config.prompt_loader import PromptLoader
from src.evaluators.llm_judge.prompt_judge import PromptJudge


def build_registry(runtime_cfg, loader):
    api_key = loader.get_api_key(runtime_cfg.llm.api_key_env)

    registry = {
        "embedding": lambda: EmbeddingEvaluator(runtime_cfg.embedding.model),
        "rouge_1": lambda: RougeEvaluator("rouge1"),
        "rouge_l": lambda: RougeEvaluator("rougeL"),
        "nli_bart": lambda: NLIEvaluator(runtime_cfg.nli.model)
    }

    # Load YAML-driven prompt judges and register them (override/add)
    try:
        pl = PromptLoader("src/evaluation_prompts.yaml")
        for name, cfg in pl.get_judges().items():
            cfg["name"] = name
            # zero-arg factory capturing cfg and api_key
            registry[name] = (lambda c=cfg: PromptJudge(c, api_key))
    except FileNotFoundError:
        # no prompts file present — skip
        pass

    return registry
