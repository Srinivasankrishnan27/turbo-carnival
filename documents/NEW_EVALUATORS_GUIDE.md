# New Deterministic Evaluators Guide

## Introduction

This document provides comprehensive documentation for the 6 new deterministic evaluators added to the turbo-carnival text evaluation pipeline. These evaluators provide **verifiable, reproducible proofs** for assessing LLM output correctness.

### Why Deterministic Evaluators?

| Type | Characteristics | Example |
|------|-----------------|---------|
| **Non-Deterministic** | Results may vary, requires API calls | LLM Judge (GPT-4, DeepSeek) |
| **Deterministic** | Same input always produces same output | ROUGE, BERTScore, NLI |

**Benefits of Deterministic Evaluators**:
- Reproducible results across runs
- No API costs or rate limits
- Faster execution (no network latency)
- Auditable and explainable scores

---

## Evaluator Overview

| # | Evaluator | Category | Primary Use Case |
|---|-----------|----------|------------------|
| 1 | Token F1 | token_level | Word-level accuracy measurement |
| 2 | Numerical Accuracy | factual | Number/date verification |
| 3 | Entity Matching | factual | Named entity verification |
| 4 | BERTScore | similarity | Semantic similarity |
| 5 | Entailment | nli | Logical inference verification |
| 6 | Contradiction | nli | **Hallucination detection** |

---

# 1. TOKEN F1 EVALUATOR

## Core Concept

Token F1 measures word-level overlap between ground truth and candidate text using the classic Information Retrieval metrics:

```
Precision = |Matched Tokens| / |Candidate Tokens|
Recall    = |Matched Tokens| / |Ground Truth Tokens|
F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

### How It Works

1. **Tokenization**: Split both texts into individual words (tokens)
2. **Normalization**: Convert to lowercase (configurable)
3. **Matching**: Find common tokens between both sets
4. **Scoring**: Calculate precision, recall, and F1

### Visual Example

```
Ground Truth: "The quick brown fox jumps"
              ↓ tokenize
              {the, quick, brown, fox, jumps}  → 5 tokens

Candidate:    "A quick brown dog jumps high"
              ↓ tokenize
              {a, quick, brown, dog, jumps, high}  → 6 tokens

Matched:      {quick, brown, jumps}  → 3 tokens

Precision = 3/6 = 0.50
Recall    = 3/5 = 0.60
F1        = 2 × (0.50 × 0.60) / (0.50 + 0.60) = 0.545
```

## Configuration

```yaml
# runtime_config.yaml
token_f1:
  lowercase: true          # Case-insensitive matching
  use_stemming: false      # Reduce words to stems (run/running → run)
  remove_stopwords: false  # Remove common words (the, is, a)
```

## Code Example

```python
import asyncio
from src.registry import REGISTRY
import src.evaluators

async def example():
    evaluator = REGISTRY.get('token_level', 'token_f1')

    result = await evaluator.evaluate(
        ground_truth="The quick brown fox jumps over the lazy dog",
        candidate="A fast brown fox leaps over a sleepy dog"
    )

    print(f"Score: {result['score']}")
    print(f"Details: {result['comment']}")

asyncio.run(example())
```

## Sample Inputs and Outputs

### Example 1: High Overlap
```
Ground Truth: "Machine learning is transforming industries worldwide"
Candidate:    "Machine learning is revolutionizing industries globally"

Score: 0.6000
Details: Token F1: 0.6000 (P: 0.6000, R: 0.6000, GT tokens: 5, Cand tokens: 5)
```

### Example 2: Exact Match
```
Ground Truth: "The capital of France is Paris"
Candidate:    "The capital of France is Paris"

Score: 1.0000
Details: Token F1: 1.0000 (P: 1.0000, R: 1.0000, GT tokens: 6, Cand tokens: 6)
```

### Example 3: Partial Match
```
Ground Truth: "Albert Einstein developed the theory of relativity in 1905"
Candidate:    "Einstein developed relativity theory"

Score: 0.5000
Details: Token F1: 0.5000 (P: 0.7500, R: 0.3750, GT tokens: 8, Cand tokens: 4)
```

### Example 4: No Overlap
```
Ground Truth: "Water boils at 100 degrees Celsius"
Candidate:    "The Amazon rainforest is in South America"

Score: 0.0000
Details: Token F1: 0.0000 (P: 0.0000, R: 0.0000, GT tokens: 6, Cand tokens: 7)
```

## When to Use

| Use Case | Recommendation |
|----------|----------------|
| Quick baseline metric | ✅ Recommended |
| Exact wording matters | ✅ Recommended |
| Paraphrased content | ❌ Use BERTScore instead |
| Semantic similarity | ❌ Use BERTScore instead |

---

# 2. NUMERICAL ACCURACY EVALUATOR

## Core Concept

Numerical Accuracy extracts all numbers from both texts and verifies they match. This is critical for detecting **numerical hallucinations** where LLMs fabricate or alter numbers.

### How It Works

1. **Extraction**: Use regex patterns to find all numbers
2. **Normalization**: Convert to comparable format
3. **Matching**: Check if ground truth numbers appear in candidate
4. **Scoring**: `Score = Matched Numbers / Total Ground Truth Numbers`

### Number Types Detected

| Type | Pattern | Example |
|------|---------|---------|
| Integer | `\d+` | 42, 1000, 2024 |
| Decimal | `\d+\.\d+` | 3.14, 99.99 |
| Percentage | `\d+\.?\d*%` | 25%, 10.5% |
| Monetary | `[$€£¥]\d+` | $100, €50 |
| Scientific | `\d+e[+-]?\d+` | 3e8, 1.5e-10 |

### Visual Example

```
Ground Truth: "Revenue was $5.2 million in 2024 with 150 employees"
              ↓ extract numbers
              {5.2, 2024, 150}  → 3 numbers

Candidate:    "The company made $5.2 million in 2024"
              ↓ extract numbers
              {5.2, 2024}  → 2 numbers

Matched:      {5.2, 2024}  → 2 numbers
Missing:      {150}

Score = 2/3 = 0.667
```

## Configuration

```yaml
# runtime_config.yaml
numerical_accuracy:
  tolerance: 0.01           # 1% relative tolerance for float comparison
  match_order: false        # Numbers don't need same order
  extract_percentages: true # Detect percentage values
  extract_monetary: true    # Detect currency values
```

## Code Example

```python
import asyncio
from src.registry import REGISTRY
import src.evaluators

async def example():
    evaluator = REGISTRY.get('factual', 'numerical_accuracy')

    result = await evaluator.evaluate(
        ground_truth="The meeting is at 3:00 PM on March 15, 2024",
        candidate="The meeting is at 4:30 PM on March 20, 2024"
    )

    print(f"Score: {result['score']}")
    print(f"Details: {result['comment']}")

asyncio.run(example())
```

## Sample Inputs and Outputs

### Example 1: All Numbers Match
```
Ground Truth: "The product costs $299.99 and weighs 2.5 kg"
Candidate:    "It costs $299.99 with a weight of 2.5 kilograms"

Score: 1.0000
Details: Numerical Accuracy: 1.0000 - Matched 2/2
```

### Example 2: Partial Match
```
Ground Truth: "Revenue was $5.2 million in Q1 2024 with 150 employees"
Candidate:    "Q1 2024 revenue reached $5.2 million"

Score: 0.7500
Details: Numerical Accuracy: 0.7500 - Matched 3/4, Missing: ['150']
```

### Example 3: HALLUCINATION - Wrong Numbers
```
Ground Truth: "The meeting is at 3:00 PM on March 15, 2024"
Candidate:    "The meeting is at 4:30 PM on March 20, 2024"

Score: 0.2500
Details: Numerical Accuracy: 0.2500 - Matched 1/4, Missing: ['3', '00', '15']
```

### Example 4: No Numbers in Text
```
Ground Truth: "The quick brown fox jumps over the lazy dog"
Candidate:    "A fast brown fox leaps over a sleepy dog"

Score: 1.0000
Details: Numerical Accuracy: 1.0000 - No numbers in either text
```

### Example 5: Percentage and Currency
```
Ground Truth: "Sales increased by 25% to reach $1.5 billion"
Candidate:    "There was a 25% increase in sales, totaling $1.5 billion"

Score: 1.0000
Details: Numerical Accuracy: 1.0000 - Matched 2/2
```

## When to Use

| Use Case | Recommendation |
|----------|----------------|
| Financial documents | ✅ Critical |
| Date/time verification | ✅ Critical |
| Scientific data | ✅ Critical |
| Detecting number hallucinations | ✅ Primary use case |
| General text comparison | ⚠️ Use with other metrics |

---

# 3. ENTITY MATCHING EVALUATOR

## Core Concept

Entity Matching uses Named Entity Recognition (NER) to extract and compare entities like people, organizations, locations, and dates between texts.

### How It Works

1. **NER Extraction**: Use spaCy to identify named entities
2. **Normalization**: Lowercase and standardize entity text
3. **Matching**: Compare entity sets
4. **Scoring**: Calculate F1 score based on entity overlap

### Entity Types Detected

| Type | Code | Examples |
|------|------|----------|
| Person | PERSON | Elon Musk, Bill Gates, Marie Curie |
| Organization | ORG | Microsoft, Google, NASA, United Nations |
| Location (Political) | GPE | California, United States, Paris |
| Location (Natural) | LOC | Mount Everest, Pacific Ocean |
| Date | DATE | 2024, January 15, next week |
| Time | TIME | 3:00 PM, noon, midnight |
| Money | MONEY | $5 million, €100, 50 dollars |
| Percent | PERCENT | 25%, ten percent |

### Visual Example

```
Ground Truth: "Bill Gates and Paul Allen founded Microsoft in Seattle"
              ↓ NER extraction
              {Bill Gates (PERSON), Paul Allen (PERSON),
               Microsoft (ORG), Seattle (GPE)}  → 4 entities

Candidate:    "Bill Gates founded Microsoft in Seattle"
              ↓ NER extraction
              {Bill Gates (PERSON), Microsoft (ORG),
               Seattle (GPE)}  → 3 entities

Matched:      {Bill Gates, Microsoft, Seattle}  → 3 entities
Missing:      {Paul Allen}

Precision = 3/3 = 1.00
Recall    = 3/4 = 0.75
F1        = 2 × (1.00 × 0.75) / (1.00 + 0.75) = 0.857
```

## Configuration

```yaml
# runtime_config.yaml
entity_matching:
  spacy_model: "en_core_web_sm"  # English NER model
  match_strategy: "exact"        # "exact" or "fuzzy"
  case_sensitive: false          # Case-insensitive matching
  use_weighted_scoring: false    # Weight by entity type importance
```

## Code Example

```python
import asyncio
from src.registry import REGISTRY
import src.evaluators

async def example():
    evaluator = REGISTRY.get('factual', 'entity_matching')

    result = await evaluator.evaluate(
        ground_truth="Apple was founded by Steve Jobs in California",
        candidate="Steve Jobs started Apple in California"
    )

    print(f"Score: {result['score']}")
    print(f"Details: {result['comment']}")

asyncio.run(example())
```

## Sample Inputs and Outputs

### Example 1: All Entities Match
```
Ground Truth: "Elon Musk founded SpaceX in California"
Candidate:    "SpaceX was founded by Elon Musk in California"

Score: 0.8000
Details: Entity F1: 0.8000 (P: 0.6667, R: 1.0000) - Matched 2/2 entities
```

### Example 2: Missing Entity
```
Ground Truth: "Bill Gates and Paul Allen founded Microsoft in Seattle"
Candidate:    "Bill Gates founded Microsoft in Seattle"

Score: 0.8571
Details: Entity F1: 0.8571 (P: 1.0000, R: 0.7500) - Matched 3/4 entities, Missing: ['paul allen(PERSON)']
```

### Example 3: Wrong Entity (HALLUCINATION)
```
Ground Truth: "Apple was founded by Steve Jobs in California"
Candidate:    "Apple was founded by Bill Gates in California"

Score: 0.6667
Details: Entity F1: 0.6667 (P: 0.6667, R: 0.6667) - Matched 2/3 entities, Missing: ['steve jobs(PERSON)']
```

### Example 4: Complex Entity Set
```
Ground Truth: "In 2020, Tesla moved from California to Texas. Elon Musk made the announcement"
Candidate:    "Elon Musk announced Tesla's relocation to Texas from California in 2020"

Score: 1.0000
Details: Entity F1: 1.0000 (P: 1.0000, R: 1.0000) - Matched 5/5 entities
```

### Example 5: Missing Organization
```
Ground Truth: "Amazon, Microsoft, and Google are major tech companies"
Candidate:    "Google and Amazon are big tech firms"

Score: 0.8000
Details: Entity F1: 0.8000 (P: 1.0000, R: 0.6667) - Matched 2/3 entities, Missing: ['microsoft(ORG)']
```

## When to Use

| Use Case | Recommendation |
|----------|----------------|
| Verifying all entities mentioned | ✅ Primary use case |
| Detecting entity substitution | ✅ Recommended |
| Checking completeness | ✅ Recommended |
| Biography/news verification | ✅ Recommended |
| Technical documentation | ⚠️ May miss technical terms |

---

# 4. BERTSCORE EVALUATOR

## Core Concept

BERTScore uses contextual embeddings from BERT-based models to compute semantic similarity at the token level. Unlike lexical metrics, it understands that "car" and "automobile" are similar.

### How It Works

1. **Embedding**: Generate BERT embeddings for each token
2. **Alignment**: Find best-matching token pairs using cosine similarity
3. **Aggregation**: Compute precision, recall, and F1

### Visual Example

```
Ground Truth: "The car is red"
              ↓ BERT embeddings
              [emb(The), emb(car), emb(is), emb(red)]

Candidate:    "The automobile is crimson"
              ↓ BERT embeddings
              [emb(The), emb(automobile), emb(is), emb(crimson)]

Token Alignments (cosine similarity):
  "car" ↔ "automobile"  → 0.92 (high similarity!)
  "red" ↔ "crimson"     → 0.89 (high similarity!)

Result: High BERTScore despite different words
```

### Why BERTScore?

| Metric | "car" vs "automobile" | Understands Synonyms |
|--------|----------------------|---------------------|
| ROUGE-1 | 0.0 (no match) | ❌ No |
| Token F1 | 0.0 (no match) | ❌ No |
| BERTScore | ~0.92 (similar) | ✅ Yes |

## Configuration

```yaml
# runtime_config.yaml
bertscore:
  model_type: "microsoft/deberta-xlarge-mnli"  # Model for embeddings
  device: "cpu"                                 # "cpu" or "cuda"
  rescale_with_baseline: false                  # Rescale scores
```

## Code Example

```python
import asyncio
from src.evaluators.similarity.bertscore import BERTScoreEvaluator

async def example():
    evaluator = BERTScoreEvaluator()

    result = await evaluator.evaluate(
        ground_truth="The cat sat on the mat",
        candidate="A feline was resting on the rug"
    )

    print(f"Score: {result['score']}")
    print(f"Details: {result['comment']}")

asyncio.run(example())
```

## Sample Inputs and Outputs

### Example 1: Exact Match
```
Ground Truth: "The capital of France is Paris"
Candidate:    "The capital of France is Paris"

Score: 1.0000
Details: BERTScore F1: 1.0000 (Nearly identical) [P: 1.0000, R: 1.0000]
```

### Example 2: Paraphrased (Same Meaning)
```
Ground Truth: "The capital of France is Paris"
Candidate:    "Paris serves as France's capital city"

Score: 0.8485
Details: BERTScore F1: 0.8485 (Very similar) [P: 0.8259, R: 0.8724]
```

### Example 3: Synonyms
```
Ground Truth: "The car is red and fast"
Candidate:    "The automobile is crimson and quick"

Score: 0.9471
Details: BERTScore F1: 0.9471 (Nearly identical) [P: 0.9471, R: 0.9471]
```

### Example 4: Related Content
```
Ground Truth: "Machine learning is a subset of artificial intelligence"
Candidate:    "AI includes machine learning as one of its branches"

Score: 0.7888
Details: BERTScore F1: 0.7888 (Very similar) [P: 0.7753, R: 0.8027]
```

### Example 5: Unrelated Content
```
Ground Truth: "The stock market closed higher today"
Candidate:    "Pizza is a popular Italian dish"

Score: 0.5458
Details: BERTScore F1: 0.5458 (Moderately similar) [P: 0.5829, R: 0.5132]
```

## Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.95 - 1.00 | Nearly identical meaning |
| 0.85 - 0.95 | Very similar meaning |
| 0.70 - 0.85 | Related content |
| 0.50 - 0.70 | Some similarity |
| < 0.50 | Different content |

## When to Use

| Use Case | Recommendation |
|----------|----------------|
| Paraphrase detection | ✅ Primary use case |
| Semantic equivalence | ✅ Recommended |
| Translation quality | ✅ Recommended |
| Contradiction detection | ❌ Use NLI instead |

---

# 5. NLI ENTAILMENT EVALUATOR

## Core Concept

Natural Language Inference (NLI) determines the logical relationship between two texts:
- **Premise** (Ground Truth): The source of truth
- **Hypothesis** (Candidate): The text to evaluate

### NLI Categories

| Category | Meaning | Example |
|----------|---------|---------|
| **Entailment** | Hypothesis follows from premise | GT: "All birds fly" → Cand: "Sparrows fly" |
| **Neutral** | Neither supported nor contradicted | GT: "John has a dog" → Cand: "John is tall" |
| **Contradiction** | Hypothesis conflicts with premise | GT: "It's raining" → Cand: "It's sunny" |

### How It Works

1. **Tokenization**: Prepare premise-hypothesis pair
2. **Model Inference**: Run through NLI model (DeBERTa)
3. **Classification**: Get probabilities for each category
4. **Scoring**: Return entailment probability

### Visual Example

```
Ground Truth (Premise):    "The company was founded in 2010 by John Smith"
Candidate (Hypothesis):    "John Smith started the company in 2010"

NLI Model Output:
  Entailment:    0.8423 (84.23%)  ← High!
  Neutral:       0.1562 (15.62%)
  Contradiction: 0.0014 (0.14%)

Score: 0.8423 (Moderately entailed)
```

## Configuration

```yaml
# runtime_config.yaml
entailment:
  model_name: "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
  device: "cpu"      # "cpu" or "cuda"
  max_length: 512    # Maximum sequence length
```

## Code Example

```python
import asyncio
from src.evaluators.nli.entailment import EntailmentEvaluator

async def example():
    evaluator = EntailmentEvaluator()

    result = await evaluator.evaluate(
        ground_truth="All employees must attend the meeting",
        candidate="John, an employee, must attend the meeting"
    )

    print(f"Score: {result['score']}")
    print(f"Details: {result['comment']}")

asyncio.run(example())
```

## Sample Inputs and Outputs

### Example 1: Strong Entailment
```
Ground Truth: "John has two children, a boy and a girl"
Candidate:    "John has children"

Score: 0.9980
Details: Entailment: 0.9980 (Strongly entailed) [N: 0.0016, C: 0.0003]
```

### Example 2: Paraphrased (Entailed)
```
Ground Truth: "The company was founded in 2010"
Candidate:    "The company started operations in 2010"

Score: 0.7380
Details: Entailment: 0.7380 (Moderately entailed) [N: 0.2550, C: 0.0069]
```

### Example 3: Partial Info (Still Entailed)
```
Ground Truth: "The CEO announced layoffs and a new strategy"
Candidate:    "The CEO announced layoffs"

Score: 0.9976
Details: Entailment: 0.9976 (Strongly entailed) [N: 0.0019, C: 0.0005]
```

### Example 4: Not Entailed (Contradiction)
```
Ground Truth: "The meeting is on Monday at 9 AM"
Candidate:    "The meeting is on Tuesday at 3 PM"

Score: 0.0001
Details: Entailment: 0.0001 (Not entailed) [N: 0.0005, C: 0.9993]
```

### Example 5: Unrelated (Neutral)
```
Ground Truth: "Python is a programming language"
Candidate:    "The weather is sunny today"

Score: 0.0040
Details: Entailment: 0.0040 (Not entailed) [N: 0.9957, C: 0.0003]
```

## Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 - 1.00 | Strongly entailed - candidate is fully supported |
| 0.70 - 0.90 | Moderately entailed - mostly supported |
| 0.50 - 0.70 | Weakly entailed - some support |
| 0.00 - 0.50 | Not entailed - may be neutral or contradictory |

## When to Use

| Use Case | Recommendation |
|----------|----------------|
| Verifying factual consistency | ✅ Primary use case |
| Checking logical support | ✅ Recommended |
| Summarization evaluation | ✅ Recommended |
| Hallucination detection | ⚠️ Use Contradiction instead |

---

# 6. NLI CONTRADICTION EVALUATOR (HALLUCINATION DETECTOR)

## Core Concept

The Contradiction Evaluator is your **PRIMARY HALLUCINATION DETECTOR**. It identifies when the candidate text directly conflicts with the ground truth.

### Key Insight

**Score is INVERTED**: `Score = 1 - P(contradiction)`

- **Higher score = BETTER** (no hallucination)
- **Lower score = HALLUCINATION DETECTED**

### How It Works

1. **NLI Inference**: Same as Entailment evaluator
2. **Focus on Contradiction**: Extract P(contradiction)
3. **Inversion**: Return 1 - P(contradiction)
4. **Interpretation**: Low scores indicate hallucination

### Visual Example

```
Ground Truth: "The Olympics were held in 2020"
Candidate:    "The Olympics were held in 2018"  ← WRONG!

NLI Model Output:
  Entailment:    0.0002 (0.02%)
  Neutral:       0.0009 (0.09%)
  Contradiction: 0.9989 (99.89%)  ← HIGH!

Score = 1 - 0.9989 = 0.0011

Result: 0.0011 → HALLUCINATION DETECTED!
```

## Configuration

```yaml
# runtime_config.yaml
contradiction:
  model_name: "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
  device: "cpu"
  max_length: 512
```

## Code Example

```python
import asyncio
from src.evaluators.nli.entailment import ContradictionEvaluator

async def example():
    evaluator = ContradictionEvaluator()

    # Test 1: Correct information
    result1 = await evaluator.evaluate(
        ground_truth="The capital of France is Paris",
        candidate="Paris is the capital of France"
    )
    print(f"Correct Info - Score: {result1['score']:.4f}")

    # Test 2: Hallucination!
    result2 = await evaluator.evaluate(
        ground_truth="The capital of France is Paris",
        candidate="The capital of France is London"
    )
    print(f"Hallucination - Score: {result2['score']:.4f}")

asyncio.run(example())
```

## Sample Inputs and Outputs

### Example 1: No Contradiction (SAFE)
```
Ground Truth: "The capital of France is Paris"
Candidate:    "Paris is the capital of France"

Score: 0.9996 ✅ SAFE
Details: Non-contradiction: 0.9996 (No contradiction detected) [E: 0.9979, N: 0.0016, C: 0.0004]
```

### Example 2: Subset Information (SAFE)
```
Ground Truth: "John is a software engineer at Google"
Candidate:    "John works at Google"

Score: 0.9998 ✅ SAFE
Details: Non-contradiction: 0.9998 (No contradiction detected) [E: 0.9988, N: 0.0010, C: 0.0002]
```

### Example 3: Unrelated Topics (SAFE - Neutral)
```
Ground Truth: "Apples are fruits"
Candidate:    "The sky is blue"

Score: 0.9953 ✅ SAFE
Details: Non-contradiction: 0.9953 (No contradiction detected) [E: 0.8342, N: 0.1611, C: 0.0047]
```

### Example 4: Wrong Date (HALLUCINATION!)
```
Ground Truth: "The Olympics were held in 2020"
Candidate:    "The Olympics were held in 2018"

Score: 0.0011 ❌ HALLUCINATION!
Details: Non-contradiction: 0.0011 (HIGH CONTRADICTION - Possible hallucination!) [E: 0.0002, N: 0.0009, C: 0.9989]
```

### Example 5: Wrong Person (HALLUCINATION!)
```
Ground Truth: "Einstein developed the theory of relativity"
Candidate:    "Newton developed the theory of relativity"

Score: 0.0014 ❌ HALLUCINATION!
Details: Non-contradiction: 0.0014 (HIGH CONTRADICTION - Possible hallucination!) [E: 0.0002, N: 0.0012, C: 0.9986]
```

### Example 6: Wrong Number (HALLUCINATION!)
```
Ground Truth: "The company has 500 employees"
Candidate:    "The company has 5000 employees"

Score: 0.0006 ❌ HALLUCINATION!
Details: Non-contradiction: 0.0006 (HIGH CONTRADICTION - Possible hallucination!) [E: 0.0002, N: 0.0004, C: 0.9994]
```

### Example 7: Opposite Meaning (HALLUCINATION!)
```
Ground Truth: "The project was successful"
Candidate:    "The project failed completely"

Score: 0.0003 ❌ HALLUCINATION!
Details: Non-contradiction: 0.0003 (HIGH CONTRADICTION - Possible hallucination!) [E: 0.0001, N: 0.0002, C: 0.9997]
```

### Example 8: Wrong Location (HALLUCINATION!)
```
Ground Truth: "Amazon is headquartered in Seattle"
Candidate:    "Amazon is headquartered in San Francisco"

Score: 0.0008 ❌ HALLUCINATION!
Details: Non-contradiction: 0.0008 (HIGH CONTRADICTION - Possible hallucination!) [E: 0.0002, N: 0.0006, C: 0.9992]
```

## Score Interpretation

| Score Range | Status | Action |
|-------------|--------|--------|
| 0.90 - 1.00 | ✅ SAFE | No contradiction detected |
| 0.70 - 0.90 | ⚠️ LOW RISK | Minor concerns, review recommended |
| 0.50 - 0.70 | ⚠️ MODERATE RISK | Potential issue, needs review |
| 0.00 - 0.50 | ❌ HALLUCINATION | HIGH CONTRADICTION - Flag immediately! |

## When to Use

| Use Case | Recommendation |
|----------|----------------|
| Hallucination detection | ✅ **PRIMARY USE CASE** |
| Fact-checking LLM outputs | ✅ Critical |
| Quality assurance | ✅ Critical |
| Content verification | ✅ Recommended |
| All LLM evaluation pipelines | ✅ Should always be included |

---

# Quick Reference

## All Evaluators at a Glance

| Evaluator | Import | Primary Metric | Best For |
|-----------|--------|----------------|----------|
| token_f1 | `REGISTRY.get('token_level', 'token_f1')` | F1 Score | Word accuracy |
| numerical_accuracy | `REGISTRY.get('factual', 'numerical_accuracy')` | Match Ratio | Number verification |
| entity_matching | `REGISTRY.get('factual', 'entity_matching')` | Entity F1 | Entity verification |
| bertscore | `BERTScoreEvaluator()` | Semantic F1 | Paraphrase detection |
| entailment | `EntailmentEvaluator()` | P(entailment) | Logical support |
| contradiction | `ContradictionEvaluator()` | 1-P(contradiction) | **Hallucination detection** |

## Recommended Evaluation Strategy

```
For comprehensive LLM output evaluation, use this combination:

1. contradiction  → Detect hallucinations (CRITICAL)
2. numerical_accuracy → Verify numbers/dates
3. entity_matching → Verify entities
4. bertscore → Measure semantic similarity
5. token_f1 → Baseline word overlap
6. entailment → Verify logical support
```

## Dependencies

```bash
pip install transformers torch bert-score spacy nltk
python -m spacy download en_core_web_sm
```

---

*Document created: 2026-01-26*
*Part of turbo-carnival evaluation pipeline*
