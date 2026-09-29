# ==============================================================================
# LLM Client Factory
# Centralizes API client construction for all providers (Gemini, Groq, OpenRouter).
# Used by both Phase 1 (Batch Evaluation) and Phase 2 (Online Shadow Evaluation).
# ==============================================================================

import os
from openai import AsyncOpenAI


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
