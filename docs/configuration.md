# Configuration Guide

The entire pipeline is controlled by a single file: `config.yaml`. This allows you to toggle behaviors without editing Python code.

## 1. The Models List (Waterfall Routing)
```yaml
models:
  judge_models:
    - "gemini/gemini-3.5-flash"      # Primary Engine
    - "groq/llama3-8b-8192"          # Fallback 1
    - "openai/gpt-4o-mini"           # Fallback 2
```
This is the most critical block. If the first model fails, the system instantly switches to the second. 
- *Note:* Our custom engine parses this list directly. The `ragas` engine takes the `[0]` index (Primary) because it doesn't natively support dynamic fallbacks.

## 2. Pipeline Switches
```yaml
pipeline:
  run_evaluation: true       # Set false to skip Phase 1
  run_calibration: true      # Set false to skip Phase 2
  evaluator_type: "custom_judge"  # Options: "custom_judge", "ragas"
```
Toggle these to run specific parts of the pipeline (e.g., if you only want to test calibration).

## 3. Thresholds
```yaml
thresholds:
  max_acceptable_delta: 0.5  # Max MAE for Phase 2
```
If the difference between Human Scores and LLM Scores is greater than `0.5`, Phase 2 will flag the judge as "Uncalibrated".

## 4. API Keys
Keys are not stored in `config.yaml`. They must be placed in a `.env` file at the root:
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`
