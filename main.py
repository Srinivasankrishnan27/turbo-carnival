import asyncio
import argparse
import json
import os
import yaml

from src.pipeline import run_pipeline
from src.runtime_config import load_runtime_config

from pydantic_evals.evaluators import LLMJudge

# Import Agentic Components (only used if --agentic is passed)
try:
    from src.orchestrator import Orchestrator
    from src.meta_evaluator import MetaEvaluator
except ImportError:
    Orchestrator = None
    MetaEvaluator = None

# Optional Tracing
try:
    import logfire
    try:
        logfire.configure(send_to_logfire='if-token-present')
        logfire.configure()
        logfire.instrument_pydantic_ai()
    except Exception as e:
        print(f"[WARN] Logfire configuration failed: {e}. Tracing disabled.")
except ImportError:
    pass


async def process_single_pair(
    ground_truth,
    candidate,
    base_config,
    runtime_cfg,
    base_dir,
    config_path,
    runtime_config_path,
    use_agentic=False,
):
    print(
        f"\n--- Processing Pair ---\nGT: {ground_truth[:50]}...\nCand: {candidate[:50]}..."
    )

    active_config_path = config_path

    # --- AGENTIC PLANNING LAYER ---
    if use_agentic:
        if not Orchestrator:
             print("[ERROR] Agentic components not found. Ensure requirements_agentic.txt is installed.")
             return None

        # Setup Agent Config
        agent_cfg = runtime_cfg.get("agent_config", {})
        if not agent_cfg:
            print(
                "[ERROR] 'agent_config' section missing in runtime_config.yaml. Cannot initialize agents."
            )
            return None

        llm_kwargs = {
            "base_url": agent_cfg.get("base_url"),
            "api_key": agent_cfg.get("api_key"),
            "model_name": agent_cfg.get("model_name"),
        }

        print("[Agent] Orchestrator is planning the evaluation...")
        orchestrator = Orchestrator(**llm_kwargs)
        meta_evaluator = MetaEvaluator(**llm_kwargs) # Initialize MetaEvaluator here for potential use

        # Create temp config path early
        import uuid
        temp_config_path = os.path.join(
            base_dir, f"_temp_agent_config_{uuid.uuid4()}.yaml"
        )

        try:
            # Plan
            plan = await orchestrator.plan_evaluation(ground_truth, candidate, base_config)
            print(f"[Agent] Plan Reasoning: {plan.reasoning}")

            # Use recommended config
            recommended_config = yaml.safe_load(plan.recommended_config)
            if "evaluators" not in recommended_config:
                recommended_config = {"evaluators": recommended_config} # Ensure it's a valid config structure

            # Standard Execution
            with open(temp_config_path, "w") as f:
                yaml.dump(recommended_config, f)
            
            active_config_path = temp_config_path
            print(f"[Agent] Created temporary configuration at {active_config_path}")

        except Exception as e:
            print(f"[WARN] Agentic planning/configuration failed: {e}. Falling back to default.")
            active_config_path = config_path
    
    # --- PIPELINE EXECUTION LAYER ---
    print("[Pipeline] Running evaluation pipeline...")
    # Clean up results
    results = {}
    try:
        results = await run_pipeline(
            ground_truth, candidate, active_config_path, runtime_config_path
        )
    except Exception as e:
         print(f"[ERROR] Pipeline execution failed: {e}")
         if use_agentic and active_config_path != config_path and os.path.exists(active_config_path):
            os.remove(active_config_path)
         return None

    # --- AGENTIC SYNTHESIS LAYER ---
    if use_agentic and results:
        print("\n[Agent] Meta-Evaluator is synthesizing results...")
        try:
            meta_evaluator = MetaEvaluator(**llm_kwargs)
            synthesis = await meta_evaluator.synthesize(results)

            # Inject meta-evaluation into results
            results["meta_evaluation"] = {
                "summary": synthesis.summary,
                "verdict": synthesis.verdict,
                "confidence": synthesis.confidence,
            }
        except Exception as e:
            print(f"[ERROR] Meta-Evaluator failed: {e}")

    # Cleanup temp config if agentic
    if use_agentic and active_config_path != config_path and os.path.exists(active_config_path):
        os.remove(active_config_path)

    return results


async def main():
    parser = argparse.ArgumentParser(description="Text Evaluation Pipeline")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to JSONL input file (ground_truth, candidate)",
    )
    parser.add_argument(
        "--output_file", type=str, default="output_results.jsonl", help="Path to save results"
    )
    parser.add_argument(
        "--agentic", action="store_true", help="Enable agentic evaluation (Planning & Synthesis)"
    )
    args = parser.parse_args()

    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    runtime_config_path = os.path.join(base_dir, "runtime_config.yaml")

    # 2. Load Base Config for reference (needed for Orchestrator)
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    runtime_cfg = load_runtime_config(runtime_config_path)

    # 3. Prepare Inputs
    inputs = []
    if args.input_file:
        if not os.path.exists(args.input_file):
             print(f"[ERROR] Input file not found: {args.input_file}")
             return
        with open(args.input_file, "r") as f:
            for line in f:
                if line.strip():
                    inputs.append(json.loads(line))
    else:
        # Default example if no input file
        print("[INFO] No input file provided. Using default example.")
        inputs.append(
            {
                "ground_truth": "The Apollo 11 mission successfully landed humans on the Moon on July 20, 1969.",
                "candidate": "Humans landed on the Moon for the first time on July 20, 1969, during the successful Apollo 11 mission.",
            }
        )

    print(f"--- Starting Evaluation for {len(inputs)} items (Agentic: {args.agentic}) ---")

    results_list = []

    for item in inputs:
        gt = item.get("ground_truth")
        cand = item.get("candidate")
        if not gt or not cand:
            print("[WARN] Skipping invalid item (missing ground_truth or candidate)")
            continue

        res = await process_single_pair(
            gt,
            cand,
            base_config,
            runtime_cfg,
            base_dir,
            config_path,
            runtime_config_path,
            use_agentic=args.agentic,
        )
        if res:
            res["input"] = item
            results_list.append(res)

    # 4. Output
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results_list, f, indent=2)
        print(f"\n[Done] Results saved to {args.output_file}")
    else:
        print("\n--- Final JSON Output ---")
        print(json.dumps(results_list[0], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
