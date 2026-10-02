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

from src.llm_client import build_async_client, clean_json_text, invoke_with_fallback
from prompts.faithfulness_judge_prompt import FAITHFULNESS_JUDGE_PROMPT
from prompts.relevancy_judge_prompt import RELEVANCY_JUDGE_PROMPT


# --- Core Evaluation Logic ---------------------------------------------------

async def evaluate_single_record(semaphore, judge_models, idx, row):
    """
    Evaluates a single record for Faithfulness and Relevancy.
    Acquires a semaphore slot to respect concurrency limits.
    """
    incident_id = row.get('incident_id', f'Record-{idx+1}')
    prompt = str(row.get('user_prompt', ''))
    context = str(row.get('raw_chat_logs', ''))
    summary = str(row.get('generated_summary', ''))

    start_time = time.perf_counter()
    used_model = "Unknown"

    async with semaphore:
        # 1. Faithfulness
        f_prompt_val = FAITHFULNESS_JUDGE_PROMPT.format(contexts=context, answer=summary)
        try:
            f_resp, f_model = await invoke_with_fallback(judge_models, f_prompt_val)
            used_model = f_model
            f_content = f_resp.choices[0].message.content
            f_parsed = json.loads(clean_json_text(f_content))
            f_score = float(f_parsed.get('faithfulness_score', 0.0))
            f_just = str(f_parsed.get('faithfulness_justification') or f_parsed.get('justification') or 'No justification provided.')
        except Exception as e:
            f_score, f_just = 0.0, f"Execution error: {e}"

        # 2. Relevancy
        r_prompt_val = RELEVANCY_JUDGE_PROMPT.format(question=prompt, answer=summary)
        try:
            r_resp, r_model = await invoke_with_fallback(judge_models, r_prompt_val)
            used_model = r_model
            r_content = r_resp.choices[0].message.content
            r_parsed = json.loads(clean_json_text(r_content))
            r_score = float(r_parsed.get('relevancy_score', 0.0))
            r_just = str(r_parsed.get('relevancy_justification') or r_parsed.get('justification') or 'No justification provided.')
        except Exception as e:
            r_score, r_just = 0.0, f"Execution error: {e}"

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        return {
            "idx": idx,
            "faithfulness": f_score,
            "faithfulness_justification": f_just,
            "answer_relevancy": r_score,
            "relevancy_justification": r_just,
            "latency_ms": latency_ms,
            "judge_model": used_model
        }


async def run_batch_evaluation(df, judge_models, max_concurrency):
    """Orchestrates concurrent evaluations across the entire dataset."""
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        evaluate_single_record(semaphore, judge_models, idx, row)
        for idx, row in df.iterrows()
    ]
    return await asyncio.gather(*tasks)


# --- Public Entry Point ------------------------------------------------------

def run_custom_judge_evaluation(
    data_path: str,
    output_path: str,
    judge_models: list,
    embedding_model_name: str = "openai/text-embedding-3-small",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.0,
    max_tokens: int = 1000,
    max_concurrency: int = 15
) -> pd.DataFrame:

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    print(f"\n[INFO] Initializing Custom Prompt Judge Fallback Array: {judge_models}")
    print(f"[INFO] Evaluating {len(df)} records concurrently (Max Concurrency: {max_concurrency})...")

    start_time = time.perf_counter()
    results = asyncio.run(run_batch_evaluation(df, judge_models, max_concurrency))
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
