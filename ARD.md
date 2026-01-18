# Architecture Reference Document (ARD)

## 1. Background

The rapid adoption of Large Language Models (LLMs) has created a critical need for robust evaluation frameworks. Traditional n-gram metrics (like ROUGE or BLEU) often fail to capture semantic nuance, while human evaluation is unscalable. "LLM-as-a-Judge" offers a promising middle ground but faces challenges in latency, cost, and reliability.

This system addresses these challenges by creating a high-throughput, unified pipeline that treats evaluation as a multi-layered process. It combines the speed of deterministic logic with the reasoning capabilities of agents, allowing for scalable, deep-dive analysis of AI-generated content.

## 2. Introduction

This project implements an asynchronous, extensible text evaluation pipeline designed to assess AI-generated content against ground truth data. It supports both deterministic metrics (e.g., ROUGE, N-Gram) and LLM-based judges (e.g., faithfulness, clarity).

The system is built with modularity and high throughput in mind, leveraging Python's `asyncio` for concurrent execution and a registry pattern for easy extensibility.

## 3. System Overview

The system operates as a pipeline:
1.  **Input**: JSONL or JSON Array of (Ground Truth, Candidate) pairs.
2.  **Orchestration (Optional)**: An LLM agent analyzes the input to recommend specific evaluators.
3.  **Execution**: The pipeline runs configured evaluators in parallel.
4.  **Aggregation**: Scores are aggregated based on configured weights.
5.  **Meta-Evaluation (Optional)**: An LLM agent synthesizes the raw metrics into a qualitative verdict.
6.  **Output**: JSONL results and optional HTML reports.

## 4. Core Architecture

### 4.1 Registry Pattern
**File**: `src/registry.py`

The system uses a central registry to manage evaluators. This decouples the definition of an evaluator from its execution.

-   **Registration**: Evaluators are registered using the `@register_evaluator` decorator found in `src/registry.py` (imported via `src.utils`).
-   **Discovery**: The `EvaluatorRegistry` class maps `(category, name)` tuples to evaluator factory functions.
-   **Extensibility**: Users can add new evaluators by simply creating a class inheriting from `BaseEvaluator`, adding the decorator, and ensuring the file is imported at runtime.

### 4.2 Pipeline Execution
**File**: `src/pipeline.py`

The `run_pipeline` function is the core execution engine.
-   **Configuration-Driven**: It reads `config.yaml` to determine which evaluators to run and their weights.
-   **Asyncio**: It uses `asyncio.gather` to run all configured evaluators concurrently for a single input pair. This significantly reduces latency compared to sequential execution, especially when using network-bound LLM judges.
-   **Layered Execution**: Evaluators are grouped by category (e.g., "ngram", "llm_judge").

### 4.3 Caching Layer
**File**: `src/utils/cache.py`

The system implements a persistent caching mechanism to avoid redundant computations, which is critical for cost-saving when using paid LLM APIs.

-   **Storage**: SQLite database (`.cache_<project_name>.db`).
-   **Granularity**: Caches individual evaluator results.
-   **Key Generation**: Unique keys are generated using a SHA256 hash of a deterministic JSON payload containing:
    -   Method Name (e.g., `llm_judge.faithfulness`)
    -   Ground Truth text
    -   Candidate text
    -   Evaluator Arguments (kwargs)
-   **Isolation**: Caches are namespaced by `project_name` to prevent collisions between different experimental runs.

### 4.4 Configuration
The system uses a split configuration model:
1.  **Pipeline Config (`config.yaml`)**: Logical configuration. Defines *what* to run (metrics, weights).
2.  **Runtime Config (`runtime_config.yaml`)**: Infrastructure configuration. Defines *how* to run it (API keys, endpoints, model names). Supports environment variable substitution (e.g., `${OPENAI_API_KEY}`).

## 5. Agentic Architecture

When `--agentic` is enabled, the system wraps the standard pipeline with intelligent planning and synthesis layers.

### 5.1 Orchestrator
**File**: `src/orchestrator.py`
-   **Role**: Pre-execution planner.
-   **Workflow**:
    1.  Analyzes the semantic content of the `ground_truth` and `candidate`.
    2.  Determines the most relevant metrics (e.g., "Is this code? Use code-eval. Is this creative writing? Use creativity judge.").
    3.  Generates a *temporary* configuration override (`_temp_agent_config_*.yaml`) tailored to the specific input.

### 5.2 Meta-Evaluator
**File**: `src/meta_evaluator.py`
-   **Role**: Post-execution analyst.
-   **Workflow**:
    1.  Receives the raw quantitative scores from the pipeline.
    2.  Uses an LLM to interpret these scores in context.
    3.  Produces a natural language summary, a final verdict (pass/fail/warning), and a confidence score.

## 6. Data Flow

```mermaid
graph TD
    Input[Input Data (JSONL)] --> Orchestrator{Agentic Plan?}
    
    Orchestrator -- Yes --> Planner[Orchestrator Agent]
    Planner --> TempConfig[Temp Config]
    TempConfig --> Pipeline
    
    Orchestrator -- No --> DefaultConfig[Default Config]
    DefaultConfig --> Pipeline
    
    Pipeline[Pipeline Execution] --> Evaluator1[Evaluator A]
    Pipeline --> Evaluator2[Evaluator B]
    Pipeline --> EvaluatorN[Evaluator ...]
    
    Evaluator1 --> Aggregator
    Evaluator2 --> Aggregator
    EvaluatorN --> Aggregator
    
    Aggregator --> MetaCheck{Agentic Custom?}
    
    MetaCheck -- Yes --> MetaEval[Meta-Evaluator Agent]
    MetaEval --> FinalResult[Final Result]
    
    MetaCheck -- No --> FinalResult
```

## 7. Key Design Decisions

1.  **Asynchronous by Default**: API-based evaluators (LLMs) are slow. Blocking IO would make large-batch evaluation impractical. `asyncio` allows thousands of concurrent requests (limited only by rate limits).
2.  **Pydantic for Validation**: `pydantic` is used extensively (via `pydantic-ai` or direct models) to ensure structured outputs from LLMs and valid internal state config.
3.  **Project Isolation**: The `CacheManager` uses `project_name` to create separate SQLite databases. This ensures that experimental runs (e.g., "test_prompt_v1") do not pollute the cache of production runs or other experiments.
