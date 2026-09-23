# LLM-as-a-Judge: Complete Technical Guide

> **Project**: Automated Quality Control Pipeline for LLM-Generated IT Incident Summaries  
> **Author**: Sumanth  
> **Stack**: Python 3.9 · LangChain 0.1.16 · Ragas 0.1.7 · OpenRouter · GPT-4o-mini  
> **Last Updated**: September 2026

---

## Table of Contents

1. [Problem Statement & Motivation](#1-problem-statement--motivation)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Project Directory Structure](#3-project-directory-structure)
4. [Data Layer — Human Baseline](#4-data-layer--human-baseline)
5. [Configuration Layer](#5-configuration-layer)
6. [Evaluator 1 — Ragas (Out-of-Box Metrics)](#6-evaluator-1--ragas-out-of-box-metrics)
7. [Evaluator 2 — Custom LLM Judge (Your Own Prompts)](#7-evaluator-2--custom-llm-judge-your-own-prompts)
8. [Prompt Engineering — Faithfulness & Relevancy](#8-prompt-engineering--faithfulness--relevancy)
9. [Calibration Gate — Delta Calculation](#9-calibration-gate--delta-calculation)
10. [Pipeline Orchestration](#10-pipeline-orchestration)
11. [Scoring System — 1-to-5 Likert Scale](#11-scoring-system--1-to-5-likert-scale)
12. [OpenRouter Integration — Why & How](#12-openrouter-integration--why--how)
13. [Key Design Decisions & Evolution History](#13-key-design-decisions--evolution-history)
14. [Known Issues & Current Status](#14-known-issues--current-status)
15. [How to Run the Pipeline](#15-how-to-run-the-pipeline)
16. [Interview Talking Points](#16-interview-talking-points)

---

## 1. Problem Statement & Motivation

### The Business Problem

In enterprise IT operations, when a major incident occurs (server crash, network outage, database deadlock), engineers discuss the issue in real-time via chat (Slack, Teams, PagerDuty). After resolution, an **LLM generates a summary** of the incident from those raw chat logs.

**The critical question**: _How do you know the LLM-generated summary is accurate and not hallucinating?_

A human reading "Payment service crashed due to OOMKilled at 512MB" needs to trust that claim actually appears in the chat logs and isn't fabricated by the LLM.

### The Solution: LLM-as-a-Judge

Instead of paying human reviewers to check every summary (expensive, slow, doesn't scale), we use a **second LLM as an automated judge** to evaluate the first LLM's output against two dimensions:

| Dimension | What It Measures | Example of Failure |
|-----------|-----------------|-------------------|
| **Faithfulness** | Are all facts in the summary supported by the chat logs? (Hallucination detection) | Summary says "router was replaced" but logs say "switch was swapped" |
| **Relevancy** | Does the summary actually answer what the user asked? (Prompt alignment) | User asked "summarize the incident" but summary talks about cafeteria closures |

### The Quality Gate

The judge's scores are compared against **human annotator scores** (ground truth). If the difference (delta) between human and judge scores is within **5%**, the judge is deemed calibrated and can be trusted in production CI/CD pipelines. If the delta exceeds 5%, the pipeline **blocks deployment** until the judge prompts are tuned.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    LLM-AS-A-JUDGE PIPELINE                       │
│                                                                  │
│  ┌────────────────┐    ┌─────────────────────┐    ┌───────────┐ │
│  │ human_baseline │    │   EVALUATION PHASE   │    │CALIBRATION│ │
│  │    .csv        │───>│                      │───>│   GATE    │ │
│  │ (20 records    │    │  ┌───────────────┐   │    │           │ │
│  │  with human    │    │  │ Ragas (OOB)   │   │    │ Human vs  │ │
│  │  1-5 scores)   │    │  │    OR         │   │    │ Judge     │ │
│  │                │    │  │ Custom Judge  │   │    │ Δ ≤ 5%?   │ │
│  └────────────────┘    │  └───────────────┘   │    │           │ │
│                        │         │             │    │ YES → ✅   │ │
│                        │         ▼             │    │ NO  → ❌   │ │
│                        │  llm_eval_results     │    │           │ │
│                        │       .json           │    └───────────┘ │
│                        └─────────────────────┘                   │
│                                                                  │
│  Config: config.yaml    Models: GPT-4o-mini via OpenRouter       │
└──────────────────────────────────────────────────────────────────┘
```

**Two-Phase Pipeline:**
1. **Phase 1 — Evaluation**: Run an LLM judge on each of the 20 incident records. Score faithfulness and relevancy on a 1-to-5 scale. Save results as structured JSON.
2. **Phase 2 — Calibration Gate**: Compare the judge's scores to the human baseline using a weighted quality formula. If the delta is ≤ 5%, the judge passes; otherwise, it fails and blocks the CI/CD pipeline.

---

## 3. Project Directory Structure

```
LLM-As-A-Judge/
│
├── .env                              # API keys (git-ignored)
├── .env.example                      # Template for .env
├── .gitignore                        # Standard Python + .env ignores
├── config.yaml                       # All pipeline configuration
├── requirements.txt                  # Pinned Python dependencies
├── run_pipeline.py                   # Main entry point — orchestrates everything
│
├── data/
│   ├── human_baseline.csv            # 20 annotated incident records (ground truth)
│   └── llm_eval_results.json         # Output: Judge scores + justifications
│
├── prompts/
│   ├── __init__.py                   # Exports both prompt templates
│   ├── faithfulness_judge_prompt.py   # Custom 1-5 faithfulness rubric with few-shot examples
│   └── relevancy_judge_prompt.py      # Custom 1-5 relevancy rubric with few-shot examples
│
├── src/
│   ├── __init__.py                   # Package marker
│   ├── evaluate_ragas.py             # Evaluator Option A: Official Ragas library
│   ├── evaluate_custom_judge.py      # Evaluator Option B: Custom prompt-based judge
│   └── calibrate_delta.py            # Phase 2: Delta calculation + calibration report
│
└── .venv/                            # Virtual environment (git-ignored)
```

### Why This Structure?

The original project had all files flat in the root directory (`evaluate_ragas.py`, `delta_caliberation.py`, `human_baseline.csv`). We restructured into a modular layout to:
- Separate **concerns**: data, prompts, source code, config.
- Enable **swappable evaluators**: `evaluate_ragas.py` vs `evaluate_custom_judge.py`.
- Keep prompts in their own **dedicated files** (not embedded in evaluation code).
- Follow Python **package conventions** with `__init__.py` files.

---

## 4. Data Layer — Human Baseline

**File**: `data/human_baseline.csv`

This is the **ground truth dataset** — 20 distinct IT incident records, each manually scored by a human annotator on a 1-to-5 scale for both faithfulness and relevancy.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `incident_id` | String | Unique ID (INC001–INC020) |
| `user_prompt` | String | What the user asked the LLM to do |
| `raw_chat_logs` | String | The raw incident chat conversation (source of truth) |
| `generated_summary` | String | The LLM's output summary |
| `human_faithfulness_score` | Integer (1-5) | Human-rated factual accuracy |
| `human_relevancy_score` | Integer (1-5) | Human-rated prompt alignment |

### Dataset Design Principles

The dataset includes **intentional failure cases** to test if the judge can detect bad outputs:

| Record | What's Wrong | Human Faithfulness | Human Relevancy |
|--------|-------------|-------------------|-----------------|
| **INC002** | Summary says "replaced router" but logs say "swapped switch" — entity hallucination | 3 | 5 |
| **INC006** | Summary adds "upgraded to Kubernetes 1.28" — not in logs | 3 | 4 |
| **INC009** | **Complete fabrication**: Logs discuss BGP route withdrawal; summary talks about "power failure" and "generator batteries" | 1 | 2 |
| **INC012** | Logs say shard was repaired; summary claims "product catalog data was permanently deleted" | 2 | 3 |
| **INC015** | User asked about staging worker nodes; summary talks about "compliance standards next month" — completely off-topic | 2 | 1 |

**Score Distribution**:
- Faithfulness: 13× score 5, 2× score 4, 2× score 3, 2× score 2, 1× score 1
- Relevancy: 16× score 5, 1× score 4, 1× score 3, 1× score 2, 1× score 1

### CSV Parsing Note

The `raw_chat_logs` field contains commas inside the text (e.g., `"SRE2: Checked logs, OOMKilled..."`). Every text field is **double-quoted** to prevent pandas `ParserError: Expected 6 fields, saw 7`. This is critical — unquoted commas inside fields break CSV parsing.

---

## 5. Configuration Layer

**File**: `config.yaml`

```yaml
pipeline:
  run_evaluation: true
  run_calibration: true
  evaluator_type: "custom_judge"    # "custom_judge" or "ragas"

models:
  generator_model: "openai/gpt-3.5-turbo"
  judge_model: "openai/gpt-4o-mini"
  embedding_model: "openai/text-embedding-3-small"
  openrouter_base_url: "https://openrouter.ai/api/v1"
  temperature: 0.0
  max_tokens: 1000

thresholds:
  max_acceptable_delta: 0.05        # 5%

paths:
  input_data: "data/human_baseline.csv"
  evaluation_output: "data/llm_eval_results.json"
```

### Key Configuration Decisions

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature: 0.0` | Zero temperature | An evaluator/judge must be **deterministic** — same input should always produce the same score |
| `max_tokens: 1000` | Capped at 1000 | OpenRouter reserves credits for the full `max_tokens` value upfront. With limited credits, the default 16,384 caused HTTP 402 errors |
| `evaluator_type` | `"custom_judge"` | Switches between Ragas (OOB) and Custom Judge evaluators |
| `max_acceptable_delta: 0.05` | 5% tolerance | Industry standard — if human and judge agree within 5%, the judge is production-ready |

### Environment Variables

**File**: `.env` (git-ignored)

```
OPENROUTER_API_KEY=sk-or-v1-xxxxx
```

The API key is loaded via `python-dotenv` at the start of `run_pipeline.py`. Both `OPENROUTER_API_KEY` and `OPENAI_API_KEY` are checked as fallbacks.

---

## 6. Evaluator 1 — Ragas (Out-of-Box Metrics)

**File**: `src/evaluate_ragas.py`

This evaluator uses the **official Ragas library (v0.1.7)** — an open-source framework specifically built for evaluating Retrieval-Augmented Generation (RAG) systems.

### How It Works — Step by Step

#### Step 1: Data Mapping

Ragas requires a HuggingFace `Dataset` with a **strict schema**:

```python
df['contexts'] = df['raw_chat_logs'].apply(lambda x: [x])   # Must be List[str]
df['question'] = df['user_prompt']                            # Renamed
df['answer']   = df['generated_summary']                      # Renamed
```

- `contexts` must be a **list of strings** (not a plain string), because RAG systems can retrieve multiple context chunks. We wrap each chat log in a list: `[x]`.
- `question` and `answer` are Ragas's expected column names.

#### Step 2: LLM + Embeddings Initialization

```python
langchain_llm = ChatOpenAI(
    model=judge_model_name,           # "openai/gpt-4o-mini"
    temperature=0.0,
    max_tokens=1000,
    openai_api_key=api_key,
    openai_api_base=openrouter_base_url,  # Redirects to OpenRouter
    default_headers=headers
)
```

Even though we use **OpenRouter** (not native OpenAI), we use LangChain's `ChatOpenAI` class because OpenRouter exposes an **OpenAI-compatible API**. The `openai_api_base` parameter redirects all API calls from `api.openai.com` to `openrouter.ai/api/v1`.

Similarly, `OpenAIEmbeddings` is initialized for `text-embedding-3-small` routed through OpenRouter.

#### Step 3: Ragas Adapter Wrappers

```python
eval_llm = LangchainLLMWrapper(langchain_llm)
eval_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
```

Ragas has its own internal interface (`BaseRagasLLM`, `BaseRagasEmbeddings`). These wrappers act as **adapters** — they take a LangChain object and translate its methods into Ragas's expected interface. Without these wrappers, Ragas cannot call the LLM.

#### Step 4: Metric Configuration

```python
faithfulness.llm = eval_llm
answer_relevancy.llm = eval_llm
answer_relevancy.embeddings = eval_embeddings
```

Ragas metrics are singleton objects. By default, they try to use OpenAI directly. We override their `.llm` and `.embeddings` attributes to point to our OpenRouter-routed models.

#### Step 5: Execution

```python
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=eval_llm,
    embeddings=eval_embeddings
)
```

This is the engine. It loops through all 20 records × 2 metrics = **40 evaluation tasks** (shown as `40/40` in the progress bar).

#### Step 6: Likert Scale Conversion

Ragas outputs continuous scores between **0.0 and 1.0**. Our system uses a **1-to-5 Likert scale**. The conversion formula:

$$\text{Score}_{1\text{-}5} = \text{round}(\text{Score}_{0\text{-}1} \times 4 + 1)$$

| Ragas Score | × 4 + 1 | Rounded | Meaning |
|-------------|---------|---------|---------|
| 1.00 | 5.0 | **5** | Perfect |
| 0.75 | 4.0 | **4** | Good |
| 0.50 | 3.0 | **3** | Partial |
| 0.25 | 2.0 | **2** | Poor |
| 0.00 | 1.0 | **1** | Failed |

#### Step 7: JSON Output with NumpyEncoder

Ragas stores `contexts` as **numpy arrays** (`np.ndarray`). Python's `json.dump()` cannot serialize numpy types natively, so we use a custom encoder:

```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)
```

### How Ragas Metrics Work Internally

#### Faithfulness (Works Well ✅)

Ragas faithfulness uses **Natural Language Inference (NLI)** — a 2-step process:

1. **Claim Extraction**: The LLM extracts individual factual claims from the summary.
   - Input: _"Payment service crashed due to OOMKilled at 512MB. Resolved by increasing to 1GB."_
   - Claims: `["Payment service crashed", "Cause was OOMKilled", "Memory limit was 512MB", "Increased to 1GB"]`

2. **Claim Verification**: Each claim is checked against the `contexts` (chat logs) using NLI.
   - For each claim: _"Does the context support this claim? (Yes/No)"_
   - Score = `(number of supported claims) / (total claims)`

This works well because it's a **logical factual check** — it doesn't depend on the phrasing of the original prompt.

#### Answer Relevancy (Broken for Summarization ❌)

Ragas relevancy uses **reverse question generation + embedding cosine similarity**:

1. **Reverse Question Generation**: Given the summary, the LLM generates plausible questions that the summary could answer.
   - Summary: _"Kafka broker-3 reached 98% disk capacity"_
   - Generated questions: `["What disk issue occurred on Kafka?", "Why was broker-3 full?"]`

2. **Embedding Comparison**: Compute cosine similarity between the generated questions and the original `user_prompt`.
   - Original prompt: _"Summarize the major incident"_ (generic)
   - Generated question: _"What disk issue occurred on Kafka?"_ (specific)
   - Cosine similarity: **~0.25** (very low)

3. **Score = average cosine similarity** across generated questions.

**Why this fails for our use case**: The `user_prompt` is always a generic instruction like "Summarize the major incident". The reverse-generated questions are always specific. Generic vs specific text has low embedding similarity, so every record scores ~0.25 → mapped to **2** on the Likert scale.

> [!IMPORTANT]
> Ragas `answer_relevancy` is designed for **specific Q&A systems** (e.g., "What is the capital of France?"), NOT for generic summarization prompts. This is a fundamental architectural mismatch, not a code bug.

---

## 7. Evaluator 2 — Custom LLM Judge (Your Own Prompts)

**File**: `src/evaluate_custom_judge.py`

This evaluator was built to solve the two critical limitations of Ragas:
1. Ragas **cannot provide textual justifications** — only bare numbers.
2. Ragas relevancy **doesn't work** for summarization tasks.

### How It Works — Step by Step

#### Step 1: Initialize LLM + Embeddings

Same pattern as Ragas — `ChatOpenAI` and `OpenAIEmbeddings` routed through OpenRouter. No Ragas wrappers needed here because we call the LLM directly.

#### Step 2: Loop Through Each Record

```python
for idx, row in df.iterrows():
    prompt = str(row.get('user_prompt', ''))
    context = str(row.get('raw_chat_logs', ''))
    summary = str(row.get('generated_summary', ''))
```

Unlike Ragas (which processes the entire dataset at once through `evaluate()`), the custom judge processes **one record at a time**, sending each to the LLM with the appropriate prompt template.

#### Step 3: Faithfulness Evaluation

```python
f_prompt_val = FAITHFULNESS_JUDGE_PROMPT.format(contexts=context, answer=summary)
f_resp = judge_llm.invoke(f_prompt_val)
f_parsed = json.loads(clean_json_text(f_resp.content))
f_score = float(f_parsed.get('faithfulness_score', 5))
f_just = str(f_parsed.get('justification', 'No justification provided.'))
```

- **Formats** the faithfulness prompt template with the chat logs and summary.
- **Invokes** the LLM (GPT-4o-mini via OpenRouter).
- **Parses** the JSON response to extract both the score AND the justification.
- The `clean_json_text()` helper strips markdown code block wrappers (`` ```json ... ``` ``) that LLMs sometimes add around their JSON output.

#### Step 4: Relevancy Evaluation

```python
r_prompt_val = RELEVANCY_JUDGE_PROMPT.format(question=prompt, answer=summary)
r_resp = judge_llm.invoke(r_prompt_val)
r_parsed = json.loads(clean_json_text(r_resp.content))
r_score = float(r_parsed.get('relevancy_score', 5))
r_just = str(r_parsed.get('justification', 'No justification provided.'))
```

Same pattern, but using the **relevancy prompt template** and passing the `user_prompt` (question) instead of the chat logs.

**Key difference from Ragas**: The custom judge directly asks the LLM "Does this summary address the user's request? Score 1-5 and explain why." No reverse question generation, no embedding math. This gives accurate relevancy scores for summarization.

#### Step 5: Embedding Similarity (Secondary Signal)

```python
p_emb = embeddings_model.embed_query(prompt)
s_emb = embeddings_model.embed_query(summary)
sim = cosine_similarity(p_emb, s_emb)
```

This computes a **supplementary** cosine similarity score between the user prompt and the summary using OpenAI embeddings. This is NOT the primary score — it's an additional reference signal stored alongside the judge's rubric-based score.

#### Step 6: Output

The custom judge outputs a JSON file with **all original columns plus**:
- `faithfulness` — 1-5 integer score
- `faithfulness_justification` — text explanation
- `answer_relevancy` — 1-5 integer score
- `relevancy_justification` — text explanation
- `embedding_similarity` — float (0.0 to 1.0)

### Ragas vs Custom Judge — Comparison

| Feature | Ragas (`evaluate_ragas.py`) | Custom Judge (`evaluate_custom_judge.py`) |
|---------|---------------------------|------------------------------------------|
| Provides justifications | ❌ No (numbers only) | ✅ Yes (JSON with reasoning) |
| Uses your custom prompts | ❌ No (Ragas's internal hidden prompts) | ✅ Yes (your rubric files in `prompts/`) |
| Faithfulness accuracy | ✅ Good (NLI-based) | ✅ Good (rubric-based) |
| Relevancy accuracy | ❌ Broken for summarization | ✅ Works correctly |
| Scale | 0.0–1.0 (converted to 1-5) | Native 1-5 |
| Speed | Faster (batch processing) | Slower (per-record API calls) |
| API calls per record | ~2 (batch optimized) | 4 (2 LLM + 2 embedding calls) |

---

## 8. Prompt Engineering — Faithfulness & Relevancy

### Faithfulness Prompt

**File**: `prompts/faithfulness_judge_prompt.py`

This prompt instructs the LLM to act as a hallucination detection specialist.

#### Structure:
1. **Role Assignment**: "You are an expert Machine Learning Quality Assurance Evaluator specializing in hallucination detection."
2. **Task Definition**: Evaluate faithfulness STRICTLY based on chat logs.
3. **Scoring Rubric** (1-5 Likert):
   - **5**: 100% factually accurate. All entities, timestamps, error states match.
   - **4**: Accurate in critical details, minor ambiguous wording.
   - **3**: Minor invented entities or swapped terms (e.g., "router" vs "switch").
   - **2**: Multiple claims conflict with logs.
   - **1**: Complete fabrication.
4. **Anti-Verbosity Rule**: "Do NOT penalize concise answers." — Prevents the judge from docking points for short but accurate summaries.
5. **Few-Shot Examples**: Three worked examples showing exact input → output JSON, including:
   - A perfect score (5) for a faithful summary
   - A partial score (3) for an entity swap
   - A score (5) for valid domain entailment ("restarted VPN to restore access")
6. **Output Schema**: Forces structured JSON: `{"justification": "...", "faithfulness_score": N}`

#### Input Variables:
- `{contexts}` — raw chat logs
- `{answer}` — generated summary

### Relevancy Prompt

**File**: `prompts/relevancy_judge_prompt.py`

This prompt instructs the LLM to evaluate prompt alignment.

#### Structure:
Same 6-part structure as faithfulness, but the rubric evaluates whether the summary **addresses the user's request**:
- **5**: Directly, accurately, and comprehensively addresses the prompt.
- **1**: Completely irrelevant (e.g., talks about cafeteria closures when asked about an incident).

#### Input Variables:
- `{question}` — user's original prompt
- `{answer}` — generated summary

### Why Separate Files?

Originally, both prompts were combined in a single file (`custom_judge_rubric.py`). They were split because:
1. **Different evaluation dimensions** require different input variables (`contexts` vs `question`).
2. Easier to **iterate and tune** each prompt independently.
3. Cleaner code — each evaluator function formats only the prompt it needs.

---

## 9. Calibration Gate — Delta Calculation

**File**: `src/calibrate_delta.py`

This is the **quality gate** that determines whether the LLM judge can be trusted in production.

### Weighted Quality Score Formula

Instead of simple averaging, we use a **weighted normalized quality score**:

$$\text{Score} = \frac{(\text{count of 5s} \times 1.0) + (\text{count of 4s} \times 0.75) + (\text{count of 3s} \times 0.50) + (\text{count of 2s} \times 0.25) + (\text{count of 1s} \times 0.0)}{\text{Total Records}}$$

| Rating | Weight | Rationale |
|--------|--------|-----------|
| 5 | 1.00 | Perfect — full credit |
| 4 | 0.75 | Good — 75% credit |
| 3 | 0.50 | Partial — half credit |
| 2 | 0.25 | Poor — quarter credit |
| 1 | 0.00 | Failed — zero credit |

**Example**: If 20 records have scores `[5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,3,3,2,2,1]`:
- Score = (13×1.0 + 2×0.75 + 2×0.50 + 2×0.25 + 1×0.0) / 20 = 16.0/20 = **0.80 (80%)**

### Calibration Delta

The **delta** is the absolute difference between the human weighted score and the LLM judge weighted score:

$$\Delta = |\text{Human Score} - \text{LLM Judge Score}|$$

If $\Delta \leq 0.05$ (5%), the judge **passes** calibration. Otherwise, it **fails**.

### Additional Metrics

The calibration report also computes:
- **Raw MAE (Mean Absolute Error)** on the 1-5 scale using scikit-learn's `mean_absolute_error`.
- **Per-Record Comparison Table** showing side-by-side human vs judge scores for every incident.
- **Audit Log** of LLM judge justifications (when using the custom judge evaluator).

### Pass/Fail Decision

```python
if faithfulness_weighted_delta > max_delta or relevancy_weighted_delta > max_delta:
    return False   # FAILED — blocks CI/CD
else:
    return True    # PASSED — judge certified for production
```

Both metrics must independently pass. If either one exceeds the 5% threshold, the entire pipeline fails.

---

## 10. Pipeline Orchestration

**File**: `run_pipeline.py`

This is the main entry point that ties everything together.

### Execution Flow

```python
def main():
    load_dotenv()                    # Load .env (API keys)
    config = yaml.safe_load(...)     # Load config.yaml
    
    # Phase 1: Evaluation
    run_evaluation(...)              # Currently calls evaluate_ragas.py
    
    # Phase 2: Calibration
    human_df = pd.read_csv(...)      # Load human baseline
    eval_df = pd.read_json(...)      # Load judge results
    passed = calculate_calibration_delta(human_df, eval_df, max_delta)
    
    if not passed:
        sys.exit(1)                  # Block CI/CD pipeline
```

### Current Wiring Status

> [!WARNING]
> As of now, `run_pipeline.py` **hardcodes** the import to use Ragas:
> ```python
> from src.evaluate_ragas import run_evaluation
> ```
> Even though `config.yaml` has `evaluator_type: "custom_judge"`, the pipeline does **not** read this setting to choose between evaluators. The custom judge (`evaluate_custom_judge.py`) exists as a complete, working script but is **not wired into the pipeline** yet.

**To use the custom judge**, `run_pipeline.py` needs to be updated to check `config['pipeline']['evaluator_type']` and conditionally import either `evaluate_ragas.run_evaluation` or `evaluate_custom_judge.run_custom_judge_evaluation`.

---

## 11. Scoring System — 1-to-5 Likert Scale

### Why 1-to-5 Instead of 0.0-1.0?

The original Ragas output uses a continuous 0.0–1.0 scale. We switched to 1-to-5 because:

1. **Human interpretability**: A score of "4" is immediately understood as "good but not perfect". A score of "0.72" requires mental conversion.
2. **Calibration alignment**: Human annotators naturally score on discrete Likert scales (1-5, 1-10). Using the same scale for both human and judge scores simplifies delta calculation.
3. **Interview context**: The 1-to-5 Likert scale is a standard in NLP evaluation literature (SummEval, G-Eval, etc.).

### Conversion Formula

For Ragas scores (0.0–1.0) → Likert (1–5):

$$\text{Likert} = \text{round}(\text{Ragas} \times 4 + 1)$$

For the custom judge, no conversion is needed — it directly outputs 1-5 integers via the prompt rubric.

---

## 12. OpenRouter Integration — Why & How

### Why OpenRouter Instead of Native OpenAI?

OpenRouter is an **API aggregator** that provides access to models from OpenAI, Anthropic, Google, Meta, and others through a **single unified API**. Benefits:
- **Single API key** for multiple model providers.
- **Pay-as-you-go** with flexible credit system.
- **OpenAI-compatible API** — same request/response format as OpenAI, so LangChain's `ChatOpenAI` works without modification.

### How the Routing Works

```
Your Code                    OpenRouter                   Model Provider
────────                    ──────────                   ──────────────
ChatOpenAI(                 
  model="openai/gpt-4o-mini",
  openai_api_base=           ──────────►
    "https://openrouter.ai/   Routes to                  ──────────►
     api/v1"                  correct                    OpenAI API
)                             provider                   (GPT-4o-mini)
                              based on                   ◄──────────
                              model slug                 Response
                             ◄──────────                
```

### Required Headers

```python
headers = {
    "HTTP-Referer": "https://github.com/Sumanth/LLM-As-A-Judge",
    "X-Title": "LLM As A Judge Pipeline"
}
```

OpenRouter uses these headers to identify your application in their dashboard. `HTTP-Referer` is recommended; `X-Title` is optional but helpful for tracking.

### Credit Constraints

OpenRouter reserves credits for the full `max_tokens` value before processing a request. With limited credits, setting `max_tokens: 16384` (default) caused:
```
HTTP 402: Could only reserve 4000 tokens, but 16384 were requested
```
Solution: Set `max_tokens: 1000` in `config.yaml`, which is sufficient for evaluation responses.

---

## 13. Key Design Decisions & Evolution History

This section documents every significant change made during development and **why** each decision was taken.

### Change 1: Flat Layout → Modular Architecture

| Before | After | Why |
|--------|-------|-----|
| `evaluate_ragas.py` (root) | `src/evaluate_ragas.py` | Separation of concerns |
| `delta_caliberation.py` (root) | `src/calibrate_delta.py` | Fixed typo + organized |
| `human_baseline.csv` (root) | `data/human_baseline.csv` | Data files in `data/` |
| `Judge Prompts/` | `prompts/` | Standard Python package |

### Change 2: OpenAI Direct → OpenRouter

**Problem**: Required specific OpenAI API key; inflexible model choice.  
**Solution**: Rerouted all API calls through OpenRouter using `openai_api_base` parameter.  
**Impact**: Can now switch between models (GPT-4o, GPT-4o-mini, Claude, Llama) by changing a single string in `config.yaml`.

### Change 3: 0/0.5/1.0 Scale → 1-to-5 Likert Scale

**Problem**: Original scoring used a 3-point scale (0, 0.5, 1.0).  
**User Requirement**: "I want the scoring metric to be 1 to 5."  
**Impact**: Updated human baseline scores, added Ragas conversion formula, rewrote calibration logic.

### Change 4: CSV Output → JSON Output

**Problem**: CSV output with commas inside text fields caused parsing errors.  
**User Requirement**: "I want the output in JSON."  
**Impact**: Changed output format to structured JSON with `indent=2`. Required `NumpyEncoder` to handle numpy arrays from Ragas.

### Change 5: Single Combined Prompt → Two Dedicated Prompt Files

**Problem**: One file tried to handle both faithfulness and relevancy in a single prompt.  
**User Requirement**: "I need different files for relevancy and faithfulness. Individually. Don't club them."  
**Impact**: Created `faithfulness_judge_prompt.py` and `relevancy_judge_prompt.py` with separate rubrics, few-shot examples, and input variables.

### Change 6: No Justifications → Mandatory JSON Justifications

**Problem**: Ragas only outputs numbers with zero explanation.  
**User Requirement**: "It should give justification for its scoring."  
**Impact**: Custom judge prompts force the LLM to output `{"justification": "...", "score": N}`. The calibration report includes an "Audit Log" section printing all justifications.

### Change 7: GPT-4o → GPT-4o-mini

**Problem**: GPT-4o with `max_tokens: 16384` caused HTTP 402 (insufficient credits).  
**Solution**: Switched to GPT-4o-mini and capped `max_tokens: 1000`.

### Change 8: ndarray JSON Crash Fix

**Problem**: Ragas stores `contexts` as numpy arrays. `json.dump()` can't serialize `ndarray`.  
**Solution**: Added `NumpyEncoder` class with handlers for `ndarray`, `np.integer`, and `np.floating`.

---

## 14. Known Issues & Current Status

### Issue 1: Pipeline Not Wired to Custom Judge (Critical)

**Status**: `run_pipeline.py` currently hardcodes `from src.evaluate_ragas import run_evaluation`.  
**Impact**: The custom judge (`evaluate_custom_judge.py`), your custom prompts, and justification output are **not being used** when you run `python run_pipeline.py`.  
**Fix Needed**: Update `run_pipeline.py` to check `config.yaml`'s `evaluator_type` and conditionally import the correct evaluator.

### Issue 2: Ragas Relevancy Scores All ~2

**Status**: Known architectural limitation.  
**Root Cause**: Ragas `answer_relevancy` uses reverse question generation + embedding similarity, which fails with generic summarization prompts.  
**Impact**: Relevancy calibration delta = 63.75% (fails the 5% gate).  
**Fix**: Use the custom judge evaluator for relevancy (once wired in).

### Issue 3: No Label Leakage

**Status**: Confirmed safe.  
**Detail**: The judge is **blind** to human scores. Neither `evaluate_ragas.py` nor `evaluate_custom_judge.py` passes `human_faithfulness_score` or `human_relevancy_score` to the LLM. Only `user_prompt`, `raw_chat_logs`, and `generated_summary` are sent.

### Last Successful Run Results (Ragas Evaluator)

| Metric | Human | LLM Judge | Delta | Status |
|--------|-------|-----------|-------|--------|
| Faithfulness | 80.00% | 81.25% | 1.25% | ✅ PASSED |
| Relevancy | 87.50% | 23.75% | 63.75% | ❌ FAILED |

---

## 15. How to Run the Pipeline

### Prerequisites

```bash
# 1. Navigate to the project
cd /Users/admin/Sumanth_projects/LLM-As-A-Judge

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API key
cp .env.example .env
# Edit .env and add your OpenRouter API key:
#   OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Running

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the full pipeline
python run_pipeline.py
```

### Expected Output

```
============================================================
      STARTING LLM-AS-A-JUDGE QUALITY CONTROL PIPELINE
============================================================

>>> Phase 1: Initializing LLM-as-a-Judge Evaluation...
[INFO] Initializing Judge Model via OpenRouter: openai/gpt-4o-mini (temp=0.0)
[INFO] Evaluating 20 records with RAGAS (faithfulness, answer_relevancy)...
Evaluating: 100%|██████████████████████████████| 40/40 [00:22<00:00, 1.82it/s]
[SUCCESS] Evaluation complete. Results saved as structured JSON to: data/llm_eval_results.json

>>> Phase 2: Initializing Delta Calibration Gate...
=================================================================
        CALIBRATION PHASE REPORT (1-to-5 Likert Metric)
=================================================================
...
```

---

## 16. Interview Talking Points

### "Walk me through the architecture."

> "We built a two-phase quality control pipeline. Phase 1 uses an LLM-as-a-Judge to evaluate IT incident summaries on faithfulness (hallucination detection) and relevancy (prompt alignment), scoring each on a 1-to-5 Likert scale. Phase 2 calibrates the judge against human annotations using a weighted quality score formula, with a 5% delta threshold as the CI/CD gate."

### "Why two evaluators?"

> "We started with Ragas, an open-source RAG evaluation framework. Ragas faithfulness worked well (1.25% delta) because it uses NLI-based claim verification. But Ragas relevancy failed catastrophically (63% delta) because its reverse question generation + embedding similarity approach doesn't work with generic summarization prompts. So we built a custom judge that directly asks the LLM to evaluate using a structured rubric, which also gives us textual justifications that Ragas cannot provide."

### "How does the faithfulness metric work?"

> "Two approaches. Ragas uses NLI: it extracts factual claims from the summary, then verifies each claim against the source chat logs. Our custom judge uses a prompt-based rubric where we directly instruct the LLM to check factual accuracy on a 1-5 scale with evidence-based justification, including few-shot examples and an anti-verbosity rule."

### "What's the anti-verbosity rule?"

> "We explicitly tell the judge: 'Do NOT penalize concise answers.' Without this, LLM judges tend to dock points for short summaries, even when they're 100% accurate. This is a known bias in LLM evaluators."

### "How do you prevent label leakage?"

> "The judge is completely blind to human scores. We only pass three fields to the evaluation: the user prompt, the raw chat logs, and the generated summary. Human faithfulness and relevancy scores exist only in the baseline CSV and are used exclusively during calibration — they never enter the judge's prompt."

### "Why 1-to-5 instead of continuous scoring?"

> "Three reasons. First, human annotators naturally think in discrete Likert scales, making calibration alignment simpler. Second, it's the standard in NLP evaluation literature (SummEval, G-Eval). Third, our weighted quality formula maps cleanly to discrete buckets."

### "What would you do differently?"

> "I'd wire the custom judge into the main pipeline from the start, add async batching to reduce latency on large datasets, implement inter-annotator agreement (Cohen's Kappa) between the human and the judge as a secondary calibration metric, and add prompt versioning to track how rubric changes affect scores over time."

---

> **End of Technical Guide**
