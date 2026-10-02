import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

# Add the project root to the Python path so we can import from 'src'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import pandas as pd
from dotenv import load_dotenv
from src.phase_2_calibration.calibration import calculate_calibration_delta


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load configuration
    config_path = PROJECT_ROOT / "config.yaml"
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"[ERROR] config.yaml not found at {config_path}")
        sys.exit(1)

    judge_type = config['pipeline']['evaluator_type']
    
    # Resolve data paths relative to project root
    input_data_path = str(PROJECT_ROOT / config['paths']['input_data'])
    output_data_path = str(PROJECT_ROOT / config['paths']['evaluation_output'])
    
    # Support both new judge_models list and old judge_model string
    judge_models = config['models'].get('judge_models')
    if not judge_models:
        judge_models = [config['models'].get('judge_model', 'gemini/gemini-3.5-flash')]
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
            from src.phase_1_evaluation.custom_judges import run_custom_judge_evaluation
            run_custom_judge_evaluation(
                data_path=input_data_path,
                output_path=output_data_path,
                judge_models=judge_models,
                embedding_model_name=embedding_model,
                openrouter_base_url=openrouter_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                max_concurrency=1
            )
        else:
            print("    Engine: Ragas Library (OOB Metrics)")
            from src.phase_1_evaluation.ragas import run_evaluation
            run_evaluation(
                data_path=input_data_path,
                output_path=output_data_path,
                judge_model_name=judge_models[2],
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
    obs_backend = online_cfg.get('observability_backend', 'native')

    # Build backend-specific kwargs from config
    backend_kwargs = {}
    if obs_backend == 'native':
        db_path = str(PROJECT_ROOT / online_cfg.get('db_path', 'data/shadow_evals.db'))
        backend_kwargs['db_path'] = db_path
        backend_kwargs['drift_threshold'] = online_cfg.get('drift_alert_threshold', 2.0)
    elif obs_backend == 'langsmith':
        backend_kwargs['project_name'] = online_cfg.get('langsmith_project', 'LLM-As-A-Judge')

    if online_cfg.get('run_simulation', False):
        print(">>> Phase 3: Simulating Online Shadow Evaluations...")
        from src.phase_3_online_evals.langsmith_online_evals import simulate_shadow_evals, run_drift_report

        sim_data_path = str(PROJECT_ROOT / online_cfg.get('simulation_data', input_data_path))

        simulate_shadow_evals(
            data_path=sim_data_path,
            judge_models=judge_models,
            sample_rate=online_cfg.get('sample_rate', 1.0),
            max_records=online_cfg.get('max_simulation_records', 10),
            drift_threshold=online_cfg.get('drift_alert_threshold', 2.0),
            observability_backend=obs_backend,
            **backend_kwargs,
        )

        run_drift_report(
            days=online_cfg.get('report_lookback_days', 7),
            observability_backend=obs_backend,
            **backend_kwargs,
        )

    elif online_cfg.get('run_drift_report', False):
        print("\n>>> Phase 3: Generating Drift Monitoring Report...")
        from src.phase_3_online_evals.langsmith_online_evals import run_drift_report
        run_drift_report(
            days=online_cfg.get('report_lookback_days', 7),
            observability_backend=obs_backend,
            **backend_kwargs,
        )

    print("=" * 60)
    print("      PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()