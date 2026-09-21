import warnings
warnings.filterwarnings("ignore")
import sys
import yaml
import pandas as pd
from dotenv import load_dotenv
from src.evaluate_ragas import run_evaluation
from src.calibrate_delta import calculate_calibration_delta

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Load configuration
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("[ERROR] config.yaml not found. Please ensure it exists in the project root.")
        sys.exit(1)

    judge_type = config['pipeline']['evaluator_type']
    input_data_path = config['paths']['input_data']
    output_data_path = config['paths']['evaluation_output']
    judge_model = config['models']['judge_model']
    embedding_model = config['models'].get('embedding_model', 'openai/text-embedding-3-small')
    openrouter_base_url = config['models'].get('openrouter_base_url', 'https://openrouter.ai/api/v1')
    temperature = config['models'].get('temperature', 0.0)
    max_tokens = config['models'].get('max_tokens', 1000)
    max_delta = config['thresholds']['max_acceptable_delta']
    
    print("=" * 60)
    print("      STARTING LLM-AS-A-JUDGE QUALITY CONTROL PIPELINE")
    print("=" * 60)
    
    # Phase 1: Evaluation with Ragas & OpenRouter
    if config['pipeline'].get('run_evaluation', True):
        print("\n>>> Phase 1: Initializing LLM-as-a-Judge Evaluation...")
        run_evaluation(
            data_path=input_data_path,
            output_path=output_data_path,
            judge_model_name=judge_model,
            embedding_model_name=embedding_model,
            openrouter_base_url=openrouter_base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        print("\n[SKIP] Skipping Evaluation Phase (pipeline.run_evaluation is False)")
        
    # Phase 2: Calibration Gate vs Human Baseline
    if config['pipeline'].get('run_calibration', True):
        print("\n>>> Phase 2: Initializing Delta Calibration Gate...")
        human_df = pd.read_csv(input_data_path)
        
        if output_data_path.endswith('.json'):
            eval_df = pd.read_json(output_data_path)
        else:
            eval_df = pd.read_csv(output_data_path)
        
        passed = calculate_calibration_delta(human_df, eval_df, max_delta)
        if not passed:
            print("\n[BLOCKED] CI/CD Pipeline Blocked: LLM Judge failed calibration gate.")
            sys.exit(1)
        else:
            print("[SUCCESS] Pipeline Complete: LLM Judge verified and certified for production.")

if __name__ == "__main__":
    main()