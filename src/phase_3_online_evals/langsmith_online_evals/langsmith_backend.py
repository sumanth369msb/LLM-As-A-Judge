# ==============================================================================
# LangSmith Observability Backend — Cloud Tracing
#
# Logs shadow evaluation results as top-level chain runs in LangSmith.
# Each evaluation creates a trace containing:
#   - Input:  The incident_id, context, and prompt
#   - Output: Faithfulness/Relevancy scores, justifications, latency, and judge model
#
# Usage:
#   Set observability_backend: "langsmith" in config.yaml
#   Requires LANGSMITH_API_KEY in .env
#
# View traces at: https://smith.langchain.com
# ==============================================================================

import os
from datetime import datetime

from .base import ObservabilityBackend


class LangSmithBackend(ObservabilityBackend):
    """
    LangSmith cloud-based observability backend.
    Creates structured trace runs for each shadow evaluation,
    enabling filtering, debugging, and drift analysis in the LangSmith UI.
    """

    def __init__(self, project_name: str = "LLM-As-A-Judge", **kwargs):
        self.project_name = project_name

        # Lazy import: only fails if the user actually selects "langsmith" backend
        try:
            from langsmith import Client
            self.client = Client()
        except ImportError:
            raise ImportError(
                "LangSmith SDK not installed. Run: pip install langsmith"
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize LangSmith client. "
                f"Ensure LANGSMITH_API_KEY is set in .env. Error: {e}"
            )

    def log_evaluation(
        self,
        incident_id: str,
        judge_model: str,
        faithfulness_score: float,
        faithfulness_justification: str,
        relevancy_score: float,
        relevancy_justification: str,
        latency_ms: float,
    ) -> None:
        """
        Creates a top-level 'chain' run in LangSmith for this evaluation.
        The run contains structured inputs and outputs for easy filtering.
        """
        self.client.create_run(
            name=f"shadow_eval_{incident_id}",
            run_type="chain",
            project_name=self.project_name,
            inputs={
                "incident_id": incident_id,
                "judge_model": judge_model,
            },
            outputs={
                "faithfulness_score": faithfulness_score,
                "faithfulness_justification": faithfulness_justification,
                "relevancy_score": relevancy_score,
                "relevancy_justification": relevancy_justification,
                "latency_ms": latency_ms,
            },
            tags=[
                f"model:{judge_model}",
                f"faith:{faithfulness_score}",
                f"relev:{relevancy_score}",
            ],
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
        )

    def generate_report(self, days: int = 7) -> None:
        """
        For LangSmith, the dashboard itself is the report.
        Prints a direct link so the user can view traces and filter by tags.
        """
        print(f"\n{'=' * 60}")
        print(f"  LANGSMITH DRIFT REPORT — Last {days} Days")
        print(f"{'=' * 60}")
        print(f"  Project: {self.project_name}")
        print(f"  View traces at: https://smith.langchain.com")
        print()
        print(f"  Recommended Filters:")
        print(f"    - Tag: 'faith:0.0' → Failed evaluations")
        print(f"    - Tag: 'relev:0.0' → Failed evaluations")
        print(f"    - Sort by: latency_ms (descending) → Slow providers")
        print(f"{'=' * 60}")
        print()
