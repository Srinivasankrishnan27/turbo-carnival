# Text Evaluation Pipeline

This project provides a flexible and extensible pipeline for evaluating candidate text against a ground truth text using various evaluation metrics.

## Table of Contents

- [Text Evaluation Pipeline](#text-evaluation-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Design Philosophy](#design-philosophy)
    - [Modular and Extensible Architecture](#modular-and-extensible-architecture)
    - [Configuration-Driven](#configuration-driven)
    - [Asynchronous Execution](#asynchronous-execution)
    - [Centralized Registry](#centralized-registry)
  - [Code Structure](#code-structure)
  - [Getting Started](#getting-started)
    - [Dependencies](#dependencies)
    - [Execution](#execution)
  - [Configuration](#configuration)
    - [Pipeline Configuration (`config.yaml`)](#pipeline-configuration-configyaml)
    - [Runtime Configuration (`runtime_config.yaml`)](#runtime-configuration-runtime_configyaml)
  - [Output Format](#output-format)
  - [How to Add a New Evaluator](#how-to-add-a-new-evaluator)
  - [Contributing](#contributing)
  - [License](#license)

## Design Philosophy

The evaluation pipeline is built on a set of core principles to ensure it is robust, flexible, and easy to maintain.

### Modular and Extensible Architecture

The core of the pipeline is the concept of "evaluators." Each evaluator is a self-contained unit responsible for a single metric (e.g., ROUGE score, semantic similarity, LLM-based judging). This design allows for new evaluation metrics to be added without modifying the core pipeline logic. Evaluators are organized into categories (e.g., `ngram`, `semantic`, `llm_judge`) for better organization.

### Configuration-Driven

The pipeline is entirely driven by configuration files. This means you can control which evaluators are run, their parameters, and their weights without changing any code.

- **`config.yaml`**: Defines the structure of the evaluation pipeline, specifying which evaluators to run for each category and their contribution (weight) to the final score.
- **`runtime_config.yaml`**: Provides a mechanism to inject runtime parameters into evaluators, such as model names, API keys, or other settings that might change between runs.

### Asynchronous Execution

To handle potentially slow, model-based evaluators, the pipeline is built using Python's `asyncio`. This allows all evaluators to run concurrently, significantly reducing the total evaluation time, especially when dealing with I/O-bound or network-bound operations (like calling an external API).

### Centralized Registry

The pipeline uses a registry pattern (`EvaluatorRegistry`) for discovering and instantiating evaluators. When the application starts, all available evaluators are registered with a central registry. The pipeline then uses this registry to look up and create instances of the evaluators specified in the `config.yaml` file. This decouples the pipeline logic from the individual evaluator implementations.

## Code Structure

```
├───.gitignore
├───config.yaml
├───LICENSE
├───main.py
├───README.md
├───requirements.txt
├───runtime_config.yaml
└───src\
    ├───__init__.py
    ├───aggregator.py
    ├───base.py
    ├───pipeline.py
    ├───registry_decorator.py
    ├───registry.py
    ├───runtime_config.py
    ├───evaluators\
    │   └───...
    └───utils\
        └───...
```

- **`main.py`**: The main entry point for running the evaluation pipeline.
- **`src/pipeline.py`**: Contains the core logic for orchestrating the evaluation pipeline.
- **`src/base.py`**: Defines the `BaseEvaluator` abstract base class that all evaluators must inherit from.
- **`src/registry.py`**: Implements the `EvaluatorRegistry`.
- **`src/registry_decorator.py`**: Provides the `@register_evaluator` decorator.
- **`src/aggregator.py`**: Responsible for aggregating scores from multiple evaluators.
- **`src/evaluators/`**: Contains the implementations of the various evaluators.
- **`src/utils/`**: Contains utility functions like the `timeit` decorator.

## Getting Started

### Dependencies

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Set up your environment variables:

```bash
cp .env.example .env
# Edit .env and add your API keys (OPENAI_API_KEY, DEEPSEEK_API_KEY, etc.)
```

### Testing

The project is equipped with a `pytest` suite for unit testing.

```bash
# Install test dependencies (included in requirements.txt)
# pip install pytest pytest-asyncio

# Run the test suite
python -m pytest tests/
```

2. Set up your environment variables:

```bash
cp .env.example .env
# Edit .env and add your API keys (e.g., OPENAI_API_KEY, DEEPSEEK_API_KEY)
```

### Execution
To run the pipeline, execute the `main.py` script:
```bash
# Basic Run
python main.py

# With HTML Report and Project Isolation
python main.py --report --project_name my_project_A

# With Agentic Mode
python main.py --agentic --report

# Batch Processing (JSONL or JSON Array)
python main.py --input_file data/sample_batch.json --output_file results.json --report
```

## Configuration

### Pipeline Configuration (`config.yaml`)

This file defines the evaluators to be used and their weights.

```yaml
evaluators:
  ngram:
    rouge1: 1.0
    rouge2: 1.0
    rougeL: 1.0
    rougeLsum: 1.0
  semantic:
    embedding: 1.0
  llm_judge:
    coverage: 1.0
    faithfulness: 1.0
    clarity: 1.0
    coherence: 1.0
    relevance: 1.0
```

### Runtime Configuration (`runtime_config.yaml`)

This file provides runtime parameters for the evaluators.

```yaml
embedding:
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  api_key : "${OPENAI_API_KEY}"
  embedding_model_name: "gemini-embedding-001"

coverage: 
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  api_key : "${OPENAI_API_KEY}"
  model_name: "gemini-2.5-flash"
  stream: False


# ... other LLM judge configurations (faithfulness, clarity, etc.)

agent_config:
  base_url: "https://api.deepseek.com"
  api_key : "${DEEPSEEK_API_KEY}"
  model_name: "deepseek:deepseek-chat"
```

## Agentic Evaluation (New)

The pipeline now supports an **agentic mode** that uses LLMs to dynamically plan the evaluation and synthesize the results.

### Execution

To run the agentic pipeline, use the `main.py` script with the `--agentic` flag:

```bash
python main.py --agentic
```

To run in batch mode using a JSONL file:

```bash
python main.py --input_file data/sample_batch.jsonl --output_file results.json --agentic
```

This mode:
1.  **Plans**: Uses an `Orchestrator` agent to analyze the input text and generate a custom configuration (e.g., prioritizing code metrics for code snippets).
2.  **Executes**: Runs the standard pipeline with the custom configuration.
3.  **Synthesizes**: Uses a `MetaEvaluator` agent to generate a qualitative report and verdict.

The behavior of these agents can be customized by editing **`prompts.yaml`**.

## Observability (Logfire)

This project integrates [pydantic-logfire](https://github.com/pydantic/logfire) for rich, structured tracing of agentic workflows.

**Setup**:
1. Run `logfire auth` to authenticate.
2. The pipeline will automatically verify your session. 
**Benefits**:
- Visualize execution traces.
- Inspect full prompt/response payloads.
- Debug latency and errors in real-time.

## Key Features

### 1. Robust Caching (`--project_name`)
The pipeline uses SQLite-based caching to prevent re-evaluating identical inputs. 
- **Isolation**: Use `--project_name <name>` to create separate cache databases (e.g., `.cache_<name>.db`) for different experiments.
- **Control**: Disable caching with `--no_cache`.

### 2. HTML Reporting (`--report`)
Generate beautiful, standalone HTML reports alongside your JSON output.
- **Dashboard**: Global scores and layer-level performance.
- **Visuals**: Interactive bar charts and color-coded indicators.
- **Detail View**: Sortable table comparing Ground Truth vs Candidate.

### 3. Flexible Input
Support for both **JSONL** (Line-delimited JSON) and **JSON Array** formats.

## Output Format

The pipeline outputs a JSON object with the aggregated scores for each category, as well as the individual scores from each evaluator.

```json
{
    "ngram": {
        "final_score": 0.85,
        "evaluators": {
            "rouge1": {
                "score": 0.9,
                "comment": "..."
            },
            "rouge2": {
                "score": 0.8,
                "comment": "..."
            }
        },
        "weights": {
            "rouge1": 0.5,
            "rouge2": 0.5
        }
    },
    "semantic": {
        ...
    },
    "llm_judge": {
        "final_score": 0.9,
        "evaluators": {
             "coverage": {
                 "score": 1.0,
                 "reason": "The candidate text covers all key points...",
                 "comment": "..."
             },
             ...
        },
        "weights": {
            "coverage": 1.0,
            ...
        }
    }
}
```

## How to Add a New Evaluator

1.  **Create the Evaluator File**: Create a new Python file in a relevant subdirectory of `src/evaluators/`.
2.  **Implement the Evaluator Class**: Define a class that inherits from `src.base.BaseEvaluator` and implements an `async` `evaluate` method.
3.  **Register the Evaluator**: Use the `@register_evaluator` decorator on a factory function for your evaluator class.
4.  **Import the Module**: Import the new evaluator module in `src/evaluators/__init__.py`.
5.  **Configure the Evaluator**: Add the new evaluator to `config.yaml` and, if needed, `runtime_config.yaml`.

For a detailed example, see the "How to Add a New Evaluator" section in the previous version of this README.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the terms of the [MIT License](LICENSE).