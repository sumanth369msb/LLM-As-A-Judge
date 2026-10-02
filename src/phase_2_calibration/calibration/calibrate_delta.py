import pandas as pd
from sklearn.metrics import mean_absolute_error

def compute_weighted_score(scores: pd.Series) -> float:
    """
    Computes the Weighted Normalized Quality Score across 1-to-5 Likert ratings:
    Formula:
      Score = (# of 5's*1.0 + # of 4's*0.75 + # of 3's*0.5 + # of 2's*0.25 + # of 1's*0.0) / total
    """
    total = len(scores)
    if total == 0:
        return 0.0
    
    # Round to nearest integer in case scores are continuous floats
    rounded = scores.round().astype(int)
    
    count_5 = (rounded == 5).sum()
    count_4 = (rounded == 4).sum()
    count_3 = (rounded == 3).sum()
    count_2 = (rounded == 2).sum()
    count_1 = (rounded == 1).sum()
    
    weighted_sum = (count_5 * 1.0) + (count_4 * 0.75) + (count_3 * 0.50) + (count_2 * 0.25) + (count_1 * 0.0)
    return weighted_sum / total


def calculate_calibration_delta(baseline_df: pd.DataFrame, eval_df: pd.DataFrame, max_delta: float) -> bool:
    """
    Calculates the calibration delta between Human Annotators and the LLM Judge using:
    1. Per-record discrete comparison (1-5 Likert scale)
    2. Weighted Normalized Quality Score Formula:
       Score = (# of 5's*1.0 + # of 4's*0.75 + # of 3's*0.5 + # of 2's*0.25) / total
    
    Args:
        baseline_df: Ground truth DataFrame containing human scores (1-5 scale).
        eval_df: Evaluation DataFrame containing LLM-as-a-Judge scores (1-5 scale).
        max_delta: Maximum acceptable delta threshold (e.g., 0.05 for 5%).
        
    Returns:
        bool: True if calibration passes within max_delta; False otherwise.
    """
    df = baseline_df.copy()
    df['llm_faithfulness_score'] = eval_df['faithfulness']
    df['llm_relevancy_score'] = eval_df['answer_relevancy']
    
    # Ensure no NaN records
    df.dropna(subset=['human_faithfulness_score', 'llm_faithfulness_score',
                      'human_relevancy_score', 'llm_relevancy_score'], inplace=True)
    
    # 1. Compute Dataset-Level Weighted Normalized Scores
    human_w_faithfulness = compute_weighted_score(df['human_faithfulness_score'])
    llm_w_faithfulness = compute_weighted_score(df['llm_faithfulness_score'])
    
    human_w_relevancy = compute_weighted_score(df['human_relevancy_score'])
    llm_w_relevancy = compute_weighted_score(df['llm_relevancy_score'])
    
    # Delta on the Weighted Normalized Score [0.0 to 1.0 scale]
    faithfulness_weighted_delta = abs(human_w_faithfulness - llm_w_faithfulness)
    relevancy_weighted_delta = abs(human_w_relevancy - llm_w_relevancy)
    
    # 2. Compute Raw Mean Absolute Error (1 to 5 scale)
    raw_faithfulness_mae = mean_absolute_error(df['human_faithfulness_score'], df['llm_faithfulness_score'])
    raw_relevancy_mae = mean_absolute_error(df['human_relevancy_score'], df['llm_relevancy_score'])
    
    print("\n" + "=" * 65)
    print("        CALIBRATION PHASE REPORT (1-to-5 Likert Metric)")
    print("=" * 65)
    print(f"Total Sample Size:     {len(df)} records")
    print(f"Max Acceptable Delta:  {max_delta:.4f} ({max_delta * 100:.1f}%)")
    print("-" * 65)
    print("WEIGHTED QUALITY SCORE FORMULA:")
    print("  Score = (#5*1.0 + #4*0.75 + #3*0.5 + #2*0.25 + #1*0.0) / Total")
    print("-" * 65)
    print(f"Human Faithfulness Score:  {human_w_faithfulness:.4f} ({human_w_faithfulness * 100:.2f}%)")
    print(f"LLM Judge Faithfulness:    {llm_w_faithfulness:.4f} ({llm_w_faithfulness * 100:.2f}%)")
    print(f">> Faithfulness Delta:     {faithfulness_weighted_delta:.4f} ({faithfulness_weighted_delta * 100:.2f}%)")
    print("-" * 65)
    print(f"Human Relevancy Score:     {human_w_relevancy:.4f} ({human_w_relevancy * 100:.2f}%)")
    print(f"LLM Judge Relevancy:       {llm_w_relevancy:.4f} ({llm_w_relevancy * 100:.2f}%)")
    print(f">> Relevancy Delta:        {relevancy_weighted_delta:.4f} ({relevancy_weighted_delta * 100:.2f}%)")
    print("-" * 65)
    print(f"Raw MAE (1-5 Scale): Faithfulness = {raw_faithfulness_mae:.2f} pts | Relevancy = {raw_relevancy_mae:.2f} pts")
    print("-" * 65)
    
    # Detailed Per-Record Comparison Table
    print("\nDetailed Per-Record Score Comparison (1-5 Scale):")
    cols = ['incident_id', 'human_faithfulness_score', 'llm_faithfulness_score',
            'human_relevancy_score', 'llm_relevancy_score']
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    print("-" * 65)
    
    # Detailed Justification Log
    if 'faithfulness_justification' in eval_df.columns:
        print("\nAudit Log - LLM Judge Justifications:")
        for _, r in eval_df.iterrows():
            inc = r.get('incident_id', 'Record')
            f_s = r.get('faithfulness', 'N/A')
            f_j = r.get('faithfulness_justification', 'N/A')
            r_s = r.get('answer_relevancy', 'N/A')
            r_j = r.get('relevancy_justification', 'N/A')
            print(f"  * {inc}:")
            print(f"    - Faithfulness ({f_s}/5): {f_j}")
            print(f"    - Relevancy    ({r_s}/5): {r_j}")
        print("-" * 65)
    
    if faithfulness_weighted_delta > max_delta or relevancy_weighted_delta > max_delta:
        print(f"\n[WARNING] Calibration delta exceeds {max_delta * 100:.1f}%.")
        print("          Status: FAILED - Judge-Prompt Tuning required before production use.\n")
        return False
    
    print(f"\n[SUCCESS] LLM Judge is calibrated within the <{max_delta * 100:.1f}% delta baseline.")
    print("          Status: PASSED - Model certified for production quality assurance.\n")
    return True
