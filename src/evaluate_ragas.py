import os
import json
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle numpy types that json.dump() cannot serialize natively."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def run_evaluation(
    data_path: str,
    output_path: str,
    judge_model_name: str,
    embedding_model_name: str = "openai/text-embedding-3-small",
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> pd.DataFrame:
    """
    Executes RAGAS evaluation metrics using OpenRouter for both LLM inference and embeddings.
    Converts Ragas 0.0-1.0 continuous scores to the 1-to-5 Likert scale for calibration.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input dataset not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Map NLP variables to RAGAS strict schema requirements
    df['contexts'] = df['raw_chat_logs'].apply(lambda x: [x])
    df['question'] = df['user_prompt']
    df['answer'] = df['generated_summary']
    
    dataset = Dataset.from_pandas(df[['question', 'contexts', 'answer']])
    
    # Retrieve OpenRouter API Key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() in ("", "sk-your-api-key-here"):
        raise ValueError(
            "OpenRouter API key is missing or not set. "
            "Please add OPENROUTER_API_KEY to your .env file."
        )
    
    headers = {
        "HTTP-Referer": "https://github.com/Sumanth/LLM-As-A-Judge",
        "X-Title": "LLM As A Judge Pipeline"
    }
    
    print(f"\n[INFO] Initializing Judge Model via OpenRouter: {judge_model_name} (temp={temperature})")
    print(f"[INFO] Initializing Embedding Model via OpenRouter: {embedding_model_name}")
    
    # 1. Initialize LangChain LLM and Embeddings directed to OpenRouter endpoint
    langchain_llm = ChatOpenAI(
        model=judge_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base=openrouter_base_url,
        default_headers=headers
    )
    
    langchain_embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        openai_api_key=api_key,
        openai_api_base=openrouter_base_url,
        default_headers=headers
    )
    
    # 2. Wrap with RAGAS wrappers
    eval_llm = LangchainLLMWrapper(langchain_llm)
    eval_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    # 3. Configure RAGAS metrics
    faithfulness.llm = eval_llm
    answer_relevancy.llm = eval_llm
    answer_relevancy.embeddings = eval_embeddings
    
    print(f"[INFO] Evaluating {len(dataset)} records with RAGAS (faithfulness, answer_relevancy)...")
    
    # 4. Execute RAGAS Evaluation
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    result_df = result.to_pandas()
    
    # 5. Convert Ragas 0.0-1.0 continuous scores into 1-to-5 Likert scale
    # Formula: Score_1_to_5 = round(Score_0_to_1 * 4 + 1)
    # (1.0 -> 5, 0.75 -> 4, 0.50 -> 3, 0.25 -> 2, 0.0 -> 1)
    result_df['faithfulness_raw'] = result_df['faithfulness']
    result_df['answer_relevancy_raw'] = result_df['answer_relevancy']
    
    result_df['faithfulness'] = (result_df['faithfulness_raw'] * 4 + 1).round()
    result_df['answer_relevancy'] = (result_df['answer_relevancy_raw'] * 4 + 1).round()
    
    # Ensure destination directory exists and save as JSON or CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith('.json'):
        records = result_df.to_dict(orient='records')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"[SUCCESS] Evaluation complete. Results saved as structured JSON to: {output_path}")
    else:
        result_df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Evaluation complete. Results saved to: {output_path}")
    
    return result_df

