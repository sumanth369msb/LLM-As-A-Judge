# System Architecture

The LLM-as-a-Judge pipeline is built to solve a specific enterprise problem: **How do we trust an AI to grade another AI?** 

To solve this, the architecture is broken down into three distinct phases.

## Phase 1: Batch Evaluation (`src/phase_1_evaluation`)
This phase is responsible for running massive datasets through an LLM evaluator. 
- **Engines Available:** You can either use our tailored `custom_judges` (which scores 1-5 with justifications) or the standard `ragas` library.
- **Resilience:** Because batch evaluation hits rate limits quickly, this phase relies heavily on our `llm_client.py`'s **Waterfall Fallback Routing**. If Gemini limits us, the traffic instantly fails over to Groq.

## Phase 2: Human Calibration Gate (`src/phase_2_calibration`)
Before a Judge can monitor live production traffic, it must prove its accuracy.
- **The Math:** We calculate the **Mean Absolute Error (MAE)** between the LLM's scores and human-annotated ground-truth scores.
- **The Gate:** If the MAE exceeds a certain threshold (e.g., > 1.0 divergence on a 5-point scale), the pipeline stops. This forces the engineer to refine the prompt before trusting it in production.

## Phase 3: Online Shadow Evaluation (`src/phase_3_online_evals`)
Once calibrated, the Judge goes to work in the real world.
- **Shadow Mode:** It runs as a non-blocking background task. When your app generates a response for a user, a copy is sent to the Judge.
- **Drift Detection:** If the Judge notices that the Faithfulness or Relevancy drops below `drift_alert_threshold` (e.g., 2.0), it fires a DRIFT ALERT.

## Multi-Provider Fallback Routing
Instead of crashing on `429 Rate Limit` or `503 Overloaded`, the system uses an `asyncio` loop to iterate through a list of providers (Gemini, Groq, OpenRouter). This guarantees near 100% uptime for the evaluation engine.
