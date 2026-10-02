# ==============================================================================
# Phase 3: Online Shadow Evaluator — Orchestration Layer
#
# This module is backend-agnostic. It handles:
#   - Sampling live production traffic
#   - Executing LLM Judge evaluations asynchronously
#   - Delegating all persistence to the configured ObservabilityBackend
#
# Observability backends live in src/observability/:
#   native_backend.py    → SQLite (local, zero deps)
#   langsmith_backend.py → LangSmith (cloud tracing)
#
# To add a new backend: create a file in src/observability/ and register it.
# No changes required here.
# ==============================================================================

import os
import re
import json
import time
import random
import asyncio
from datetime import datetime

from openai import RateLimitError

from src.llm_client import build_async_client, clean_json_text, invoke_with_fallback
from src.phase_3_online_evals.custom_online_evals import get_backend, ObservabilityBackend
from prompts.faithfulness_judge_prompt import FAITHFULNESS_JUDGE_PROMPT
from prompts.relevancy_judge_prompt import RELEVANCY_JUDGE_PROMPT


# --- Core Evaluation Logic ---------------------------------------------------

async def _run_evaluation(judge_models, incident_id, context, prompt, summary):
    """
    Calls the LLM Judge for Faithfulness and Relevancy on a single record.
    Returns a dict of raw scores, justifications, and latency.
    """
    start = time.perf_counter()
    used_model = "Unknown"

    # 1. Faithfulness
    f_prompt = FAITHFULNESS_JUDGE_PROMPT.format(contexts=context, answer=summary)
    try:
        f_resp, f_model = await invoke_with_fallback(judge_models, f_prompt)
        used_model = f_model
        f_parsed = json.loads(clean_json_text(f_resp.choices[0].message.content))
        f_score = float(f_parsed.get("faithfulness_score", 0.0))
        f_just = str(f_parsed.get("faithfulness_justification", ""))
    except Exception as e:
        f_score, f_just = 0.0, f"Error: {e}"

    # 2. Relevancy
    r_prompt = RELEVANCY_JUDGE_PROMPT.format(question=prompt, answer=summary)
    try:
        r_resp, r_model = await invoke_with_fallback(judge_models, r_prompt)
        used_model = r_model
        r_parsed = json.loads(clean_json_text(r_resp.choices[0].message.content))
        r_score = float(r_parsed.get("relevancy_score", 0.0))
        r_just = str(r_parsed.get("relevancy_justification", ""))
    except Exception as e:
        r_score, r_just = 0.0, f"Error: {e}"

    return {
        "faithfulness_score": f_score,
        "faithfulness_justification": f_just,
        "relevancy_score": r_score,
        "relevancy_justification": r_just,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "judge_model": used_model
    }


# ==============================================================================
# ShadowEvaluator — Production Integration Class
# ==============================================================================

class ShadowEvaluator:
    """
    Asynchronous Shadow Evaluator for production request handlers.

    Instantiate once at app startup. Call enqueue() from your request handler.
    Evaluation happens in the background — the caller is never blocked.
    """

    def __init__(
        self,
        judge_models: list = None,
        sample_rate: float = 0.05,
        drift_threshold: float = 2.0,
        observability_backend: str = "native",
        **backend_kwargs,
    ):
        if judge_models is None:
            judge_models = ["gemini/gemini-3.5-flash"]
        elif isinstance(judge_models, str):
            judge_models = [judge_models]
            
        self.judge_models = judge_models
        self.sample_rate = sample_rate
        self.drift_threshold = drift_threshold
        self.backend: ObservabilityBackend = get_backend(observability_backend, **backend_kwargs)

    def enqueue(self, incident_id: str, context: str, prompt: str, summary: str) -> bool:
        """
        Probabilistically samples and enqueues an async background evaluation.
        Returns True if sampled, False if skipped.
        """
        if random.random() > self.sample_rate:
            return False
        asyncio.create_task(self._evaluate_and_log(incident_id, context, prompt, summary))
        return True

    async def _evaluate_and_log(self, incident_id: str, context: str, prompt: str, summary: str):
        result = await _run_evaluation(
            self.judge_models, incident_id, context, prompt, summary
        )
        judge_model = result.pop("judge_model", self.judge_models[0])
        self.backend.log_evaluation(incident_id=incident_id, judge_model=judge_model, **result)

        if result["faithfulness_score"] <= self.drift_threshold or \
           result["relevancy_score"] <= self.drift_threshold:
            print(
                f"[DRIFT ALERT] {incident_id} — "
                f"Faithfulness: {result['faithfulness_score']}/5, "
                f"Relevancy: {result['relevancy_score']}/5"
            )


# ==============================================================================
# Simulation Runner — Test Phase 3 Without a Live App
# ==============================================================================

def simulate_shadow_evals(
    data_path: str,
    judge_models: list = None,
    sample_rate: float = 1.0,
    max_records: int = 10,
    drift_threshold: float = 2.0,
    observability_backend: str = "native",
    **backend_kwargs,
):
    """
    Reads records from a CSV and evaluates them as if they were live traffic.
    Useful for testing Phase 3 without deploying a production application.

    Args:
        data_path:              CSV file with incident records.
        judge_models:           List of models for waterfall fallback routing.
        sample_rate:            Fraction to evaluate (1.0 = all, for testing).
        max_records:            Max records to process in this simulation run.
        drift_threshold:        Score at or below which a DRIFT ALERT is emitted.
        observability_backend:  "native" or "langsmith".
        **backend_kwargs:       Forwarded to the backend constructor.
    """
    import pandas as pd

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path).head(max_records)
    
    if judge_models is None:
        judge_models = ["gemini/gemini-3.5-flash"]
    elif isinstance(judge_models, str):
        judge_models = [judge_models]

    backend: ObservabilityBackend = get_backend(observability_backend, **backend_kwargs)
    backend_label = "LangSmith Cloud" if observability_backend == "langsmith" else "SQLite (Native)"

    print(f"\n[PHASE 3] Simulating Online Shadow Evaluation...")
    print(f"  Judge Models:   {judge_models}")
    print(f"  Backend:        {backend_label}")
    print(f"  Sample Rate:    {sample_rate * 100:.0f}%")
    print(f"  Records:        {len(df)}")
    print()

    async def _run():
        for idx, row in df.iterrows():
            if random.random() > sample_rate:
                continue

            incident_id = row.get("incident_id", f"SIM-{idx+1}")
            context = str(row.get("raw_chat_logs", ""))
            prompt = str(row.get("user_prompt", ""))
            summary = str(row.get("generated_summary", ""))

            print(f"  Evaluating {incident_id}...", end=" ", flush=True)

            result = await _run_evaluation(judge_models, incident_id, context, prompt, summary)
            
            judge_model = result.pop("judge_model", judge_models[0])
            backend.log_evaluation(incident_id=incident_id, judge_model=judge_model, **result)

            f_score = result["faithfulness_score"]
            r_score = result["relevancy_score"]
            print(f"Faith: {f_score}/5 | Relev: {r_score}/5 | {result['latency_ms']:.0f}ms ({judge_model})")

            if f_score <= drift_threshold or r_score <= drift_threshold:
                print(f"    [DRIFT ALERT] Low quality detected for {incident_id}!")

    asyncio.run(_run())
    print(f"\n[SUCCESS] Shadow evaluations saved to: {backend_label}")


# ==============================================================================
# Drift Report Entry Point
# ==============================================================================

def run_drift_report(
    days: int = 7,
    observability_backend: str = "native",
    **backend_kwargs,
):
    """
    Generates a drift report via the configured observability backend.
    For 'native': prints terminal report from SQLite.
    For 'langsmith': prints dashboard URL.
    """
    backend: ObservabilityBackend = get_backend(observability_backend, **backend_kwargs)
    backend.generate_report(days=days)
