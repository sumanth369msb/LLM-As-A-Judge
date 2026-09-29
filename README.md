# LLM-As-A-Judge Quality Control Framework

## Overview
The **LLM-As-A-Judge Quality Control Framework** is a production-ready, highly configurable pipeline designed to automate the evaluation of LLM-generated outputs (like summaries or RAG responses). By employing a secondary LLM as a "Judge", this framework assesses generations for faithfulness and relevancy, seamlessly comparing them against human-annotated baselines. 

Built with CI/CD integration in mind, the pipeline incorporates a **Calibration Gate** to automatically block model deployments if the LLM Judge's scoring deviates beyond a maximum acceptable delta from the human baseline. 

## Key Architectural Features

- **3-Phase Pipeline**: Batch Evaluation → Calibration Gate → Online Shadow Evaluation
- **Pluggable Evaluation Engines**: Choose between custom Judge prompts (1-5 scale with chain-of-thought justifications) or standard `ragas` metrics via a config toggle.
- **Multi-Provider Support**: Seamlessly swap between Google Gemini, Groq, OpenRouter, and OpenAI with a single config change. Zero code modifications required.
- **Automated CI/CD Calibration Gate**: Phase 2 calculates the Weighted Delta between the LLM Judge's scores and the human baseline. If the delta exceeds the configurable threshold, the pipeline fails with a non-zero exit code.
- **Online Shadow Evaluation**: Phase 3 samples live production traffic, evaluates it asynchronously in the background, and monitors for quality drift over time.
- **Config-Driven Design**: 100% controlled via `config.yaml`.

## Project Structure

```
├── config.yaml                   # Central configuration for all 3 phases
├── run_pipeline.py               # Main entry point orchestrating the full pipeline
├── requirements.txt              # Python dependencies
├── .env                          # API keys (GEMINI_API_KEY, GROQ_API_KEY, etc.)
│
├── src/
│   ├── llm_client.py             # Centralized LLM client factory (Gemini/Groq/OpenRouter)
│   ├── evaluate_custom_judge.py  # Phase 1: Batch evaluation with custom judge prompts
│   ├── evaluate_ragas.py         # Phase 1: Alternative evaluation using Ragas library
│   ├── calibrate_delta.py        # Phase 2: Calibration gate (LLM vs Human delta)
│   └── online_evaluator.py       # Phase 3: Shadow evaluation & drift monitoring
│
├── prompts/                      # Modular prompt templates (LangChain-free)
│   ├── __init__.py
│   ├── faithfulness_judge_prompt.py
│   └── relevancy_judge_prompt.py
│
├── data/
│   ├── human_baseline.csv        # Human-annotated ground truth (20 records)
│   ├── load_test_150.csv         # Extended load test dataset (150 records)
│   ├── llm_eval_results.json     # Phase 1 output
│   └── shadow_evals.db           # Phase 3 SQLite observability database
│
└── docs/
    ├── LLM_As_A_Judge.md
    └── SCALING_AND_PRODUCTION_ROADMAP.md
```

## How It Works

### Phase 1: Batch LLM-as-a-Judge Evaluation
Reads the dataset, sends each record through the configured LLM Judge for Faithfulness and Relevancy scoring (1-5 Likert scale with chain-of-thought justifications), and saves the results.

### Phase 2: Calibration Gate vs Human Baseline
Mathematically compares the LLM Judge scores against human-annotated scores using a Weighted Normalized Quality Score. If the delta exceeds the threshold (default: 5%), the pipeline fails — acting as an automated CI/CD safety gate.

### Phase 3: Online Shadow Evaluation & Drift Monitoring
Samples a configurable percentage of live production traffic, evaluates it asynchronously in the background using the same Judge prompts, and persists results to an SQLite database. A drift report CLI detects quality regressions over time.

**Simulation Mode**: For testing without a live app, Phase 3 can simulate shadow evaluations from a CSV dataset.

**Production Integration**: The `ShadowEvaluator` class is designed to be instantiated once at app startup and called from any request handler (FastAPI, Flask, etc.):

```python
from src.online_evaluator import ShadowEvaluator

evaluator = ShadowEvaluator(judge_model="gemini/gemini-3.8-flash", sample_rate=0.05)

# Inside your request handler:
evaluator.enqueue(incident_id, context, prompt, summary)  # Non-blocking
```

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables** — Create a `.env` file:
   ```env
   GEMINI_API_KEY=your_google_ai_studio_key
   GROQ_API_KEY=your_groq_key        # Optional
   OPENROUTER_API_KEY=your_key       # Optional
   ```

3. **Run the Pipeline:**
   ```bash
   # Run Phase 1 (Evaluation) + Phase 2 (Calibration Gate)
   python run_pipeline.py

   # Run Phase 3 Simulation (Shadow Eval on 10 records + Drift Report)
   # Set online_evaluation.run_simulation: true in config.yaml, then:
   python run_pipeline.py
   ```

## Supported LLM Providers

| Provider | Config Prefix | Free Tier | Example |
| :--- | :--- | :--- | :--- |
| Google Gemini | `gemini/` | ✅ 15 RPM, 1M TPM | `gemini/gemini-3.8-flash` |
| Groq | `groq/` | ✅ 14,400 RPM, 8K TPM | `groq/qwen/qwen3.8-27b` |
| OpenRouter | *(none)* | ✅ 50 req/day | `meta-llama/llama-3.3-70b-instruct:free` |
| OpenAI | *(none)* | ❌ Paid | `openai/gpt-4o-mini` |

## Documentation
- [LLM As A Judge Concept](docs/LLM_As_A_Judge.md)
- [Scaling and Production Roadmap](docs/SCALING_AND_PRODUCTION_ROADMAP.md)

---
*Designed with robustness, modularity, and automated quality control for modern GenAI engineering teams.*
