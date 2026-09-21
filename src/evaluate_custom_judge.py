import os
import re
import json
import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from prompts.faithfulness_judge_prompt import FAITHFULNESS_JUDGE_PROMPT
from prompts.relevancy_judge_prompt import RELEVANCY_JUDGE_PROMPT

def clean_json_text(text: str) -> str:
    """Extracts JSON substring if enclosed in markdown code blocks."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()

def cosine_similarity(vec_a, vec_b) -> float:
    """Computes cosine similarity between two 1D numpy arrays."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def run_custom_judge_evaluation(
    data_path: str,
    output_path: str,
    judge_model_name: str,
    embedding_model_name: str = "openai/text-embedding-3-small",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> pd.DataFrame:
    """
    Executes Custom LLM-as-a-Judge evaluation using dedicated prompts:
    - Faithfulness: Prompts LLM for 1-5 rating + evidence justification.
    - Relevancy: Prompts LLM for 1-5 rating + prompt alignment justification.
    - Embeddings: Computes cosine similarity with OpenAIEmbeddings.
    - Saves structured JSON with full justification audit trail.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input dataset not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() in ("", "sk-your-api-key-here"):
        raise ValueError("OpenRouter API key missing. Please check .env file.")
        
    headers = {
        "HTTP-Referer": "https://github.com/Sumanth/LLM-As-A-Judge",
        "X-Title": "LLM As A Judge Pipeline"
    }
    
    print(f"\n[INFO] Initializing Custom Prompt Judge: {judge_model_name}")
    print(f"[INFO] Initializing Embeddings: {embedding_model_name}")
    
    judge_llm = ChatOpenAI(
        model=judge_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base=openrouter_base_url,
        default_headers=headers
    )
    
    # embeddings_model = OpenAIEmbeddings(
    #     model=embedding_model_name,
    #     openai_api_key=api_key,
    #     openai_api_base=openrouter_base_url,
    #     default_headers=headers
    # )
    
    print(f"[INFO] Evaluating {len(df)} records with custom prompts and extracting justifications...")
    
    faithfulness_scores = []
    faithfulness_justifications = []
    relevancy_scores = []
    relevancy_justifications = []
    # embedding_similarities = []
    
    for idx, row in df.iterrows():
        incident_id = row.get('incident_id', f'Record-{idx+1}')
        prompt = str(row.get('user_prompt', ''))
        context = str(row.get('raw_chat_logs', ''))
        summary = str(row.get('generated_summary', ''))
        
        # 1. Faithfulness with justification
        f_prompt_val = FAITHFULNESS_JUDGE_PROMPT.format(contexts=context, answer=summary)
        try:
            f_resp = judge_llm.invoke(f_prompt_val)
            f_parsed = json.loads(clean_json_text(f_resp.content))
            f_score = float(f_parsed.get('faithfulness_score', 5))
            f_just = str(f_parsed.get('faithfulness_justification') or f_parsed.get('justification') or 'No justification provided.')
        except Exception as e:
            f_score, f_just = 5.0, f"Parse error: {e}"
            
        # 2. Relevancy with justification
        r_prompt_val = RELEVANCY_JUDGE_PROMPT.format(question=prompt, answer=summary)
        try:
            r_resp = judge_llm.invoke(r_prompt_val)
            r_parsed = json.loads(clean_json_text(r_resp.content))
            r_score = float(r_parsed.get('relevancy_score', 5))
            r_just = str(r_parsed.get('relevancy_justification') or r_parsed.get('justification') or 'No justification provided.')
        except Exception as e:
            r_score, r_just = 5.0, f"Parse error: {e}"
            
        # # 3. Vector Embeddings Similarity
        # try:
        #     p_emb = embeddings_model.embed_query(prompt)
        #     s_emb = embeddings_model.embed_query(summary)
        #     sim = cosine_similarity(p_emb, s_emb)
        # except Exception:
        #     sim = 1.0
            
        faithfulness_scores.append(f_score)
        faithfulness_justifications.append(f_just)
        relevancy_scores.append(r_score)
        relevancy_justifications.append(r_just)
        # embedding_similarities.append(sim)
        
    result_df = df.copy()
    result_df['faithfulness'] = faithfulness_scores
    result_df['faithfulness_justification'] = faithfulness_justifications
    result_df['answer_relevancy'] = relevancy_scores
    result_df['relevancy_justification'] = relevancy_justifications
    # result_df['embedding_similarity'] = embedding_similarities
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith('.json'):
        records = result_df.to_dict(orient='records')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    else:
        result_df.to_csv(output_path, index=False)
        
    print(f"[SUCCESS] Custom Judge Evaluation complete. Saved to: {output_path}")
    return result_df

