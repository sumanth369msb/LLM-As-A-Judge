# ==============================================================================
# Phase 2: Online Shadow Evaluator — Production Drift Monitoring
#
# Evaluates a configurable sample of live production LLM outputs in the
# background (asynchronously), logs scores to an SQLite database, and exposes
# a drift-reporting CLI to detect quality regressions over time.
#
# Architecture:
#   Production App ──► enqueue_evaluation() ──► (async background task)
#                                                  │
#                                        ┌─────────▼──────────┐
#                                        │  shadow_evals.db   │
#                                        └─────────┬──────────┘
#                                                  │
#                                        run_drift_report() ◄── CLI / Cron
# ==============================================================================

import os
import re
import json
import time
import random
import asyncio
import sqlite3
from datetime import datetime, timedelta

from openai import RateLimitError

from src.llm_client import build_async_client
from prompts.faithfulness_judge_prompt import FAITHFULNESS_JUDGE_PROMPT
from prompts.relevancy_judge_prompt import RELEVANCY_JUDGE_PROMPT


# --- Utilities (shared with batch evaluator) ----------------------------------

def _clean_json_text(text: str) -> str:
    """Extracts JSON from optional markdown code fences."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


async def _invoke_with_retry(client, model_name, prompt_val, max_retries=5):
    """Chat completion with backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_val}],
                temperature=0.0,
                max_tokens=1000
            )
        except RateLimitError as e:
            wait = 5.0
            m = re.search(r'try again in ([\d\.]+)s', str(e))
            if m:
                wait = float(m.group(1)) + 0.5
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                wait = 15.0 * (attempt + 1)
            else:
                wait = 2.0
            await asyncio.sleep(wait)


# --- Database Layer -----------------------------------------------------------

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_evals (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                     TEXT    NOT NULL,
    incident_id                   TEXT    NOT NULL,
    faithfulness_score            REAL,
    faithfulness_justification    TEXT,
    relevancy_score               REAL,
    relevancy_justification       TEXT,
    latency_ms                    REAL,
    judge_model                   TEXT
);
"""


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Returns a connection and ensures the schema exists."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(DB_SCHEMA)
    conn.commit()
    return conn


# --- Shadow Evaluator Class ---------------------------------------------------

class ShadowEvaluator:
    """
    Asynchronous Shadow Evaluator for Production Observability.

    Designed to be instantiated once at application startup and called from
    your request handler. It randomly samples a percentage of live traffic,
    evaluates the output in the background, and persists the results.

    Usage:
        evaluator = ShadowEvaluator(config)
        evaluator.enqueue(incident_id, context, prompt, summary)
    """

    def __init__(
        self,
        judge_model: str = "gemini/gemini-3.8-flash",
        sample_rate: float = 0.05,
        db_path: str = "data/shadow_evals.db",
        drift_threshold: float = 2.0,
    ):
        """
        Args:
            judge_model:      Model identifier with provider prefix (e.g. "gemini/gemini-3.8-flash").
            sample_rate:      Fraction of traffic to evaluate (0.0–1.0). Default 5%.
            db_path:          Path to the SQLite observability database.
            drift_threshold:  Score at or below which a DRIFT ALERT is emitted.
        """
        self.judge_model = judge_model
        self.sample_rate = sample_rate
        self.db_path = db_path
        self.drift_threshold = drift_threshold

        # Build the LLM client once; reused across all evaluations.
        self.client, self.real_model = build_async_client(judge_model)

        # Ensure DB schema exists.
        _get_connection(db_path).close()

    # --- Core Evaluation (runs in background) --------------------------------

    async def _evaluate_and_log(self, incident_id: str, context: str, prompt: str, summary: str):
        """Calls the Judge for both metrics and persists the results."""
        start = time.perf_counter()

        # 1. Faithfulness
        f_prompt = FAITHFULNESS_JUDGE_PROMPT.format(contexts=context, answer=summary)
        try:
            f_resp = await _invoke_with_retry(self.client, self.real_model, f_prompt)
            f_parsed = json.loads(_clean_json_text(f_resp.choices[0].message.content))
            f_score = float(f_parsed.get("faithfulness_score", 1.0))
            f_just = str(f_parsed.get("faithfulness_justification", ""))
        except Exception as e:
            f_score, f_just = 1.0, f"Error: {e}"

        # 2. Relevancy
        r_prompt = RELEVANCY_JUDGE_PROMPT.format(question=prompt, answer=summary)
        try:
            r_resp = await _invoke_with_retry(self.client, self.real_model, r_prompt)
            r_parsed = json.loads(_clean_json_text(r_resp.choices[0].message.content))
            r_score = float(r_parsed.get("relevancy_score", 1.0))
            r_just = str(r_parsed.get("relevancy_justification", ""))
        except Exception as e:
            r_score, r_just = 1.0, f"Error: {e}"

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # 3. Persist to SQLite
        with _get_connection(self.db_path) as conn:
            conn.execute(
                """INSERT INTO shadow_evals
                   (timestamp, incident_id, faithfulness_score, faithfulness_justification,
                    relevancy_score, relevancy_justification, latency_ms, judge_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.utcnow().isoformat(), incident_id,
                 f_score, f_just, r_score, r_just, latency_ms, self.judge_model)
            )

        # 4. Drift Alert
        if f_score <= self.drift_threshold or r_score <= self.drift_threshold:
            print(
                f"[DRIFT ALERT] {incident_id} — "
                f"Faithfulness: {f_score}/5, Relevancy: {r_score}/5"
            )

    # --- Public API -----------------------------------------------------------

    def enqueue(self, incident_id: str, context: str, prompt: str, summary: str) -> bool:
        """
        Probabilistically enqueues a shadow evaluation.
        Call this from your request handler — it returns immediately.

        Returns True if the record was sampled for evaluation, False if skipped.
        """
        if random.random() > self.sample_rate:
            return False

        asyncio.create_task(
            self._evaluate_and_log(incident_id, context, prompt, summary)
        )
        return True


# --- Drift Report (CLI / Cron) ------------------------------------------------

def run_drift_report(db_path: str = "data/shadow_evals.db", days: int = 7):
    """
    Generates a Drift Monitoring Report from the shadow evaluation database.
    Reads the last N days of evaluations and prints aggregate quality metrics.

    Args:
        db_path: Path to the SQLite shadow evaluations database.
        days:    Number of days to look back.
    """
    if not os.path.exists(db_path):
        print(f"[ERROR] Shadow evaluation database not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    cursor = conn.execute(
        """SELECT COUNT(*),
                  ROUND(AVG(faithfulness_score), 2),
                  ROUND(AVG(relevancy_score), 2),
                  ROUND(MIN(faithfulness_score), 1),
                  ROUND(MIN(relevancy_score), 1),
                  ROUND(AVG(latency_ms), 1),
                  SUM(CASE WHEN faithfulness_score <= 2.0 THEN 1 ELSE 0 END),
                  SUM(CASE WHEN relevancy_score <= 2.0 THEN 1 ELSE 0 END)
           FROM shadow_evals
           WHERE timestamp >= ?""",
        (cutoff,)
    )
    row = cursor.fetchone()
    total, avg_f, avg_r, min_f, min_r, avg_lat, drift_f, drift_r = row

    if total == 0:
        print(f"[INFO] No shadow evaluations recorded in the last {days} days.")
        conn.close()
        return

    print("\n" + "=" * 65)
    print("       ONLINE SHADOW EVALUATION — DRIFT MONITORING REPORT")
    print("=" * 65)
    print(f"  Reporting Period:       Last {days} days (since {cutoff[:10]})")
    print(f"  Total Records Sampled:  {total}")
    print("-" * 65)
    print(f"  Avg Faithfulness Score: {avg_f}/5.0")
    print(f"  Avg Relevancy Score:    {avg_r}/5.0")
    print(f"  Min Faithfulness Score: {min_f}/5.0")
    print(f"  Min Relevancy Score:    {min_r}/5.0")
    print("-" * 65)
    print(f"  Drift Alerts (Faith ≤ 2): {int(drift_f)} records")
    print(f"  Drift Alerts (Relev ≤ 2): {int(drift_r)} records")
    print(f"  Avg Judge Latency:        {avg_lat}ms")
    print("-" * 65)

    # Quality SLA check
    faith_ok = avg_f >= 4.0
    relev_ok = avg_r >= 4.0
    if faith_ok and relev_ok:
        print("  [SLA PASS] Production quality within acceptable thresholds.")
    else:
        if not faith_ok:
            print(f"  [SLA BREACH] Avg Faithfulness {avg_f} is below 4.0 threshold!")
        if not relev_ok:
            print(f"  [SLA BREACH] Avg Relevancy {avg_r} is below 4.0 threshold!")
    print("=" * 65 + "\n")

    # Show the 5 worst-scoring records for triage
    worst = conn.execute(
        """SELECT incident_id, faithfulness_score, relevancy_score,
                  faithfulness_justification, timestamp
           FROM shadow_evals
           WHERE timestamp >= ?
           ORDER BY (faithfulness_score + relevancy_score) ASC
           LIMIT 5""",
        (cutoff,)
    ).fetchall()

    if worst:
        print("  Lowest-Scoring Records (Candidates for Human Review):")
        print("  " + "-" * 63)
        for w in worst:
            print(f"    {w[0]} | Faith: {w[1]}/5 | Relev: {w[2]}/5 | {w[4][:10]}")
            print(f"      Justification: {w[3][:120]}...")
        print()

    conn.close()


# --- Simulation Runner (for testing without a live app) -----------------------

def simulate_shadow_evals(
    data_path: str,
    judge_model: str = "gemini/gemini-3.8-flash",
    sample_rate: float = 1.0,
    db_path: str = "data/shadow_evals.db",
    max_records: int = 10,
):
    """
    Simulates online shadow evaluation by reading records from a CSV and
    running the evaluator on each one sequentially. Useful for testing the
    Phase 2 pipeline without deploying a live application.

    Args:
        data_path:    Path to CSV containing incident records.
        judge_model:  Model identifier with provider prefix.
        sample_rate:  Fraction to evaluate (1.0 = all records for testing).
        db_path:      Path to the SQLite database.
        max_records:  Maximum number of records to simulate.
    """
    import pandas as pd

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path).head(max_records)

    print(f"\n[PHASE 2] Simulating Online Shadow Evaluation...")
    print(f"  Judge Model:   {judge_model}")
    print(f"  Sample Rate:   {sample_rate * 100:.0f}%")
    print(f"  Records:       {len(df)}")
    print(f"  Database:      {db_path}\n")

    client, real_model = build_async_client(judge_model)

    async def _run():
        for idx, row in df.iterrows():
            if random.random() > sample_rate:
                continue

            incident_id = row.get("incident_id", f"SIM-{idx+1}")
            context = str(row.get("raw_chat_logs", ""))
            prompt = str(row.get("user_prompt", ""))
            summary = str(row.get("generated_summary", ""))

            print(f"  Evaluating {incident_id}...", end=" ", flush=True)
            start = time.perf_counter()

            # --- Faithfulness ---
            f_prompt = FAITHFULNESS_JUDGE_PROMPT.format(contexts=context, answer=summary)
            try:
                f_resp = await _invoke_with_retry(client, real_model, f_prompt)
                f_parsed = json.loads(_clean_json_text(f_resp.choices[0].message.content))
                f_score = float(f_parsed.get("faithfulness_score", 1.0))
                f_just = str(f_parsed.get("faithfulness_justification", ""))
            except Exception as e:
                f_score, f_just = 1.0, f"Error: {e}"

            await asyncio.sleep(4.1)  # Pacing for free-tier RPM limits

            # --- Relevancy ---
            r_prompt = RELEVANCY_JUDGE_PROMPT.format(question=prompt, answer=summary)
            try:
                r_resp = await _invoke_with_retry(client, real_model, r_prompt)
                r_parsed = json.loads(_clean_json_text(r_resp.choices[0].message.content))
                r_score = float(r_parsed.get("relevancy_score", 1.0))
                r_just = str(r_parsed.get("relevancy_justification", ""))
            except Exception as e:
                r_score, r_just = 1.0, f"Error: {e}"

            await asyncio.sleep(4.1)  # Pacing for free-tier RPM limits

            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            # Persist
            with _get_connection(db_path) as conn:
                conn.execute(
                    """INSERT INTO shadow_evals
                       (timestamp, incident_id, faithfulness_score, faithfulness_justification,
                        relevancy_score, relevancy_justification, latency_ms, judge_model)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (datetime.utcnow().isoformat(), incident_id,
                     f_score, f_just, r_score, r_just, latency_ms, judge_model)
                )

            print(f"Faith: {f_score}/5 | Relev: {r_score}/5 | {latency_ms:.0f}ms")

            if f_score <= 2.0 or r_score <= 2.0:
                print(f"    [DRIFT ALERT] Low quality detected for {incident_id}!")

    asyncio.run(_run())
    print(f"\n[SUCCESS] Shadow evaluations saved to: {db_path}")
