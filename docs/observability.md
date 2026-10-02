# Observability & Tracing

In Phase 3 (Online Shadow Evaluation), it is critical to log every evaluation so you can track how your AI is performing in production over time.

We use a **Strategy Pattern** for this, meaning you can swap where logs are sent simply by updating `config.yaml`.

## Option A: LangSmith Cloud (`langsmith`)
LangSmith is an enterprise LLMOps platform by LangChain. 
- **How it works:** When a shadow evaluation completes, it sends the prompt, the context, the scores (Faithfulness/Relevancy), and the latency to your LangSmith dashboard.
- **Why use it:** It gives you a beautiful UI to see exact trace-level execution, filter by low scores, and monitor token usage.

## Option B: Native SQLite (`native`)
If you don't want to use a cloud service, the pipeline falls back to a local SQLite database (`data/shadow_evals.db`).
- **How it works:** It creates a table and inserts the evaluation metrics as rows. 
- **Why use it:** It is entirely local, requires zero API keys, and is perfect for local testing or strict data-privacy requirements.

## Drift Alerts
Regardless of which backend you use, the `online_evaluator.py` script checks the final scores. If an AI response gets a score lower than the `drift_alert_threshold` set in `config.yaml`, it prints a severe warning to the terminal alerting you that the production model is hallucinating or losing relevancy.
