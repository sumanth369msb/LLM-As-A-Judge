import warnings
warnings.filterwarnings("ignore")
import sys
import yaml
import pandas as pd
from dotenv import load_dotenv
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

    # ── Phase 1: Batch LLM-as-a-Judge Evaluation ─────────────────────────────
    if config['pipeline'].get('run_evaluation', True):
        print("\n>>> Phase 1: Initializing LLM-as-a-Judge Evaluation...")
        if judge_type == 'custom_judge':
            print("    Engine: Custom Judge Prompts (1-5 Likert Scale)")
            from src.evaluate_custom_judge import run_custom_judge_evaluation
            run_custom_judge_evaluation(
                data_path=input_data_path,
                output_path=output_data_path,
                judge_model_name=judge_model,
                embedding_model_name=embedding_model,
                openrouter_base_url=openrouter_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                max_concurrency=1
            )
        else:
            print("    Engine: Ragas Library (OOB Metrics)")
            from src.evaluate_ragas import run_evaluation
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

    # ── Phase 2: Calibration Gate vs Human Baseline ──────────────────────────
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
            print("[PASS] Calibration Gate Passed: LLM Judge certified for production.\n")

    # ── Phase 3: Online Shadow Evaluation (Production Drift Monitoring) ──────
    online_cfg = config.get('online_evaluation', {})
    if online_cfg.get('run_simulation', False):
        print(">>> Phase 3: Simulating Online Shadow Evaluations...")
        from src.online_evaluator import simulate_shadow_evals, run_drift_report

        simulate_shadow_evals(
            data_path=online_cfg.get('simulation_data', input_data_path),
            judge_model=judge_model,
            sample_rate=online_cfg.get('sample_rate', 1.0),
            db_path=online_cfg.get('db_path', 'data/shadow_evals.db'),
            max_records=online_cfg.get('max_simulation_records', 10),
        )

        # Generate drift report from the recorded evaluations
        run_drift_report(
            db_path=online_cfg.get('db_path', 'data/shadow_evals.db'),
            days=online_cfg.get('report_lookback_days', 7),
        )

    elif online_cfg.get('run_drift_report', False):
        print("\n>>> Phase 3: Generating Drift Monitoring Report...")
        from src.online_evaluator import run_drift_report
        run_drift_report(
            db_path=online_cfg.get('db_path', 'data/shadow_evals.db'),
            days=online_cfg.get('report_lookback_days', 7),
        )

    print("=" * 60)
    print("      PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()