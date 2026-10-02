# ==============================================================================
# LLM Client Factory
# Centralizes API client construction for all providers (Gemini, Groq, OpenRouter).
# Used by both Phase 1 (Batch Evaluation) and Phase 2 (Online Shadow Evaluation).
# ==============================================================================

import os
import re
import asyncio
from openai import AsyncOpenAI, RateLimitError


def build_async_client(judge_model_name: str, openrouter_base_url: str = "https://openrouter.ai/api/v1"):
    """
    Builds an AsyncOpenAI client and resolves the real model name
    based on the provider prefix in judge_model_name.

    Supported prefixes:
        - "gemini/"  → Google Generative AI (OpenAI-compatible endpoint)
        - "groq/"    → Groq Cloud
        - (default)  → OpenRouter or native OpenAI

    Returns:
        tuple: (client: AsyncOpenAI, real_model_name: str)
    """
    if judge_model_name.startswith("gemini/"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        real_model = judge_model_name.replace("gemini/", "")
        headers = None

    elif judge_model_name.startswith("groq/"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        base_url = "https://api.groq.com/openai/v1"
        real_model = judge_model_name.replace("groq/", "")
        headers = None

    else:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key (OPENROUTER_API_KEY or OPENAI_API_KEY) not found in .env")
        base_url = openrouter_base_url
        real_model = judge_model_name
        headers = {
            "HTTP-Referer": "https://github.com/Sumanth/LLM-As-A-Judge",
            "X-Title": "LLM As A Judge Pipeline"
        }

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=headers,
        max_retries=2
    )

    return client, real_model

# ==============================================================================
# Shared LLM Utilities
# ==============================================================================

def clean_json_text(text: str) -> str:
    """Extracts JSON from optional markdown code fences."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


async def invoke_with_fallback(judge_models: list, prompt_val: str, max_loops: int = 2):
    """
    Executes ChatCompletion using a fallback waterfall strategy.
    If the first model hits a 429 or 503, it instantly routes to the next model in the list.
    Returns: (response, successful_model_str)
    """
    if isinstance(judge_models, str):
        judge_models = [judge_models]

    for loop in range(max_loops):
        for model_str in judge_models:
            client, real_model = build_async_client(model_str)
            try:
                resp = await client.chat.completions.create(
                    model=real_model,
                    messages=[{"role": "user", "content": prompt_val}],
                    temperature=0.0,
                    max_tokens=1000
                )
                return resp, model_str
            except RateLimitError:
                print(f"      [FALLBACK] {model_str} rate limited. Switching to next provider...")
                continue
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str:
                    print(f"      [FALLBACK] {model_str} overloaded (503). Switching to next provider...")
                    continue
                else:
                    print(f"      [WARNING] {model_str} failed: {e}. Switching...")
                    continue
        
        # If we exhausted all models in the list, wait before trying the list again
        if loop < max_loops - 1:
            print("      [RATE LIMIT] All providers exhausted. Sleeping 10s before retry loop...")
            await asyncio.sleep(10.0)

    raise Exception("All fallback providers failed after max retries.")
