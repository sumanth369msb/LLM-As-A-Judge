# ==============================================================================
# Phase 1: Custom Prompt Judge — Batch Evaluation Engine
# Evaluates a full dataset of LLM-generated summaries against custom
# Faithfulness and Relevancy rubrics using an LLM-as-a-Judge approach.
# ==============================================================================

import os
import re
import json
import time
import asyncio
import numpy as np
import pandas as pd
from openai import RateLimitError

from src.llm_client import build_async_client
from prompts.faithfulness_judge_prompt import FAITHFULNESS_JUDGE_PROMPT
from prompts.relevancy_judge_prompt import RELEVANCY_JUDGE_PROMPT


# --- Utilities ---------------------------------------------------------------

def clean_json_text(text: str) -> str:
    """Extracts JSON substring if enclosed in markdown code blocks."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


async def invoke_with_retry(client, model_name, prompt_val, max_retries=10):
    """Executes ChatCompletion with robust backoff for 429 TPM/RPM limits."""
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_val}],
                temperature=0.0,
                max_tokens=1000
            )
            return resp
        except RateLimitError as e:
            err_msg = str(e)
            wait_time = 5.0
            m = re.search(r'try again in ([\d\.]+)s', err_msg)
            if m:
                wait_time = float(m.group(1)) + 0.5
            if attempt == max_retries - 1:
                raise e
            print(f"      [RATE LIMIT] Waiting {wait_time:.1f}s before retry...")
            await asyncio.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            # 503 Service Unavailable — server overloaded, wait longer
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                wait_time = 15.0 * (attempt + 1)  # 15s, 30s, 45s... progressive backoff
                print(f"      [503 UNAVAILABLE] Server overloaded. Waiting {wait_time:.0f}s before retry ({attempt+1}/{max_retries})...")
            else:
                wait_time = 2.0
                print(f"      [RETRY] Error: {e}. Retrying ({attempt+1}/{max_retries})...")
            await asyncio.sleep(wait_time)


# --- Core Evaluation Logic ---------------------------------------------------

async def evaluate_single_record(semaphore, client, model_name, idx, row):
    """
    Evaluates a single record for Faithfulness and Relevancy.
    Acquires a semaphore slot to respect concurrency limits.
    """
    incident_id = row.get('incident_id', f'Record-{idx+1}')
    prompt = str(row.get('user_prompt', ''))
    context = str(row.get('raw_chat_logs', ''))
    summary = str(row.get('generated_summary', ''))

    start_time = time.perf_counter()

    async with semaphore:
        # 1. Faithfulness
        f_prompt_val = FAITHFULNESS_JUDGE_PROMPT.format(contexts=context, answer=summary)
        try:
            f_resp = await invoke_with_retry(client, model_name, f_prompt_val)
            f_content = f_resp.choices[0].message.content
            f_parsed = json.loads(clean_json_text(f_content))
            f_score = float(f_parsed.get('faithfulness_score', 5))
            f_just = str(f_parsed.get('faithfulness_justification') or f_parsed.get('justification') or 'No justification provided.')
        except Exception as e:
            f_score, f_just = 1.0, f"Execution error: {e}"

        await asyncio.sleep(8.0)  # ~7 RPM — safely within Gemini 15 RPM free tier

        # 2. Relevancy
        r_prompt_val = RELEVANCY_JUDGE_PROMPT.format(question=prompt, answer=summary)
        try:
            r_resp = await invoke_with_retry(client, model_name, r_prompt_val)
            r_content = r_resp.choices[0].message.content
            r_parsed = json.loads(clean_json_text(r_content))
            r_score = float(r_parsed.get('relevancy_score', 5))
            r_just = str(r_parsed.get('relevancy_justification') or r_parsed.get('justification') or 'No justification provided.')
        except Exception as e:
            r_score, r_just = 1.0, f"Execution error: {e}"

        await asyncio.sleep(8.0)  # ~7 RPM — safely within Gemini 15 RPM free tier

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        return {
            "idx": idx,
            "faithfulness": f_score,
            "faithfulness_justification": f_just,
            "answer_relevancy": r_score,
            "relevancy_justification": r_just,
            "latency_ms": latency_ms
        }


async def run_batch_evaluation(df, client, model_name, max_concurrency):
    """Orchestrates concurrent evaluations across the entire dataset."""
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        evaluate_single_record(semaphore, client, model_name, idx, row)
        for idx, row in df.iterrows()
    ]
    return await asyncio.gather(*tasks)


# --- Public Entry Point ------------------------------------------------------

def run_custom_judge_evaluation(
    data_path: str,
    output_path: str,
    judge_model_name: str,
    embedding_model_name: str = "openai/text-embedding-3-small",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.0,
    max_tokens: int = 1000,
    max_concurrency: int = 15
) -> pd.DataFrame:

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    print(f"\n[INFO] Initializing Custom Prompt Judge: {judge_model_name}")

    client, real_model = build_async_client(judge_model_name, openrouter_base_url)

    print(f"[INFO] Evaluating {len(df)} records concurrently (Max Concurrency: {max_concurrency})...")

    start_time = time.perf_counter()
    results = asyncio.run(run_batch_evaluation(df, client, real_model, max_concurrency))
    total_time = time.perf_counter() - start_time

    print(f"[INFO] Batch evaluation completed in {total_time:.2f} seconds.")

    faithfulness_scores = []
    faithfulness_justifications = []
    relevancy_scores = []
    relevancy_justifications = []
    latencies = []

    for res in results:
        faithfulness_scores.append(res["faithfulness"])
        faithfulness_justifications.append(res["faithfulness_justification"])
        relevancy_scores.append(res["answer_relevancy"])
        relevancy_justifications.append(res["relevancy_justification"])
        latencies.append(res["latency_ms"])

    result_df = df.copy()
    result_df['faithfulness'] = faithfulness_scores
    result_df['faithfulness_justification'] = faithfulness_justifications
    result_df['answer_relevancy'] = relevancy_scores
    result_df['relevancy_justification'] = relevancy_justifications
    result_df['latency_ms'] = latencies

    avg_latency = np.mean(latencies)
    print(f"[INFO] Average Latency per record: {avg_latency:.2f}ms")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith('.json'):
        records_to_save = result_df.to_dict(orient='records')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records_to_save, f, indent=2, ensure_ascii=False)
    else:
        result_df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Custom Judge Evaluation complete. Saved to: {output_path}")
    return result_df
