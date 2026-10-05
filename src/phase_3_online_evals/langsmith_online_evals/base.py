# ==============================================================================
# Observability Backend — Abstract Base Class (Strategy Pattern)
#
# Defines the interface that all observability backends must implement.
# This allows the online evaluator to be completely backend-agnostic:
#   - NativeBackend    → SQLite (local, zero external deps)
#   - LangSmithBackend → LangSmith Cloud (enterprise tracing)
# ==============================================================================

from abc import ABC, abstractmethod


class ObservabilityBackend(ABC):
    """
    Abstract interface for all observability backends.
    Every backend must implement two methods:
      - log_evaluation(): Persist a single shadow evaluation result.
      - generate_report(): Generate a drift/quality summary over a time window.
    """

    @abstractmethod
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
        """Persist a single evaluation result to the backend."""
        ...

    @abstractmethod
    def generate_report(self, days: int = 7) -> None:
        """Generate and display a drift/quality report for the given lookback window."""
        ...

