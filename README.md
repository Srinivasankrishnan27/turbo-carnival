# Turbo Carnival: Text Evaluation Pipeline

A robust, extensible, and asynchronous pipeline for evaluating AI-generated text against ground truth using a suite of deterministic metrics and LLM-based judges.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Standard Evaluation](#standard-evaluation)
  - [Agentic Evaluation](#agentic-evaluation)
  - [Advanced Features](#advanced-features)
- [Configuration](#configuration)
  - [Pipeline Config](#pipeline-config)
  - [Runtime Config](#runtime-config)
- [Architecture](#architecture)
- [Contributing](#contributing)

---

## Features

- **Protocol-Agnostic LLM Judges**: Compatible with any OpenAI-compatible API (DeepSeek, Gemini, OpenAI, etc.).
- **Asynchronous Execution**: High-throughput evaluation using Python `asyncio`.
- **Agentic Workflows**:
    - **Orchestrator**: Dynamically plans evaluations based on input content.
    - **Meta-Evaluator**: Synthesizes scores into a final qualitative verdict.
- **Project Isolation**: Workspace-aware caching preventing collisions between experiments.
- **Rich Reporting**: Generates interactive HTML dashboards with charts and global metrics.
- **Flexible Inputs**: Supports both JSONL and JSON Array input formats.
- **Observability**: Built-in support for [Logfire](https://github.com/pydantic/logfire) tracing.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd turbo-carnival
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Copy the example environment file and add your API keys.
   ```bash
   cp .env.example .env
   # Edit .env to set OPENAI_API_KEY, DEEPSEEK_API_KEY, etc.
   ```

---

## Quick Start

Run the pipeline with the default sample data:

```bash
python main.py
```

This will:
1. Load sample data from the script.
2. Run configured evaluators (ROUGE, simple LLM judge).
3. Print the results to the console.

---

## Usage Guide

The `main.py` script is the primary entry point.

### Standard Evaluation
Process a batch of file inputs and save results to JSON.

```bash
python main.py \
  --input_file data/sample_batch.json \
  --output_file results.json
```

### Agentic Evaluation
Enable the **Orchestrator** and **Meta-Evaluator** agents to plan and summarize the run.

```bash
python main.py \
  --agentic \
  --input_file data/sample_batch.json \
  --output_file results_agentic.json
```

### Advanced Features

#### 1. HTML Reporting (`--report`)
Generate a visual HTML dashboard (`results_report.html`) alongside the JSON output.
```bash
python main.py --report ...
```

#### 2. Caching & Project Isolation (`--project_name`)
Prevent re-running expensive evaluations on the same input. Use a project name to separate caches (e.g., dev vs prod).
```bash
python main.py --project_name experiment_A ...
```
To force a re-run, use `--no_cache`.

#### 3. Batch Input Formats
Supported formats for `--input_file`:
*   **JSON Array** (`.json`): `[{"ground_truth": "...", "candidate": "..."}]`
*   **JSON Lines** (`.jsonl`): `{"ground_truth": "...", "candidate": "..."}\n...`

---

## Configuration

### Pipeline Config
**File**: `config.yaml`  
Defines *what* to run and *how* to weigh it.

```yaml
evaluators:
  ngram:
    rouge1: 1.0
  llm_judge:
    faithfulness: 1.0
    clarity: 1.0
```

### Runtime Config
**File**: `runtime_config.yaml`  
Defines *how* to connect (models, API keys, endpoints). Supports environment variable expansion.

```yaml
llm_judge:
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model_name: "gpt-4"

agent_config:
  base_url: "https://api.deepseek.com"
  model_name: "deepseek-chat"
```

---

## Architecture

The project follows a **Registry Pattern**:

1.  **Registry**: Evaluators are registered via `@register_evaluator` in `src/evaluators`.
2.  **Pipeline**: `src/pipeline.py` reads `config.yaml`, instantiates registered evaluators, and runs them via `asyncio.gather`.
3.  **Agents**:
    *   `src/orchestrator.py`: Analyzes input -> outputs Config Override.
    *   `src/meta_evaluator.py`: Analyzes Results -> outputs Text Summary.

---

## Contributing

1.  Create a new evaluator in `src/evaluators/`.
2.  Inherit from `BaseEvaluator`.
3.  Add the `@register_evaluator` decorator.
4.  Add the new metric to `config.yaml`.

## License

MIT License.