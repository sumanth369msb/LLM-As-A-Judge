# ==============================================================================
# Observability Backend Factory
#
# Provides a clean public API for the rest of the codebase:
#   from src.phase_3_online_evals.custom_online_evals import get_backend, ObservabilityBackend
#
# The factory function get_backend() instantiates the correct backend
# based on the string value from config.yaml (e.g., "native" or "langsmith").
# ==============================================================================

from .base import ObservabilityBackend
from .native_backend import NativeBackend
from .langsmith_backend import LangSmithBackend


def get_backend(backend_name: str = "native", **kwargs) -> ObservabilityBackend:
    """
    Factory function that returns the appropriate ObservabilityBackend instance.

    Args:
        backend_name: "native" for SQLite, "langsmith" for LangSmith cloud.
        **kwargs:     Forwarded to the backend constructor
                      (e.g., db_path for native, project_name for langsmith).

    Returns:
        An initialized ObservabilityBackend instance.

    Raises:
        ValueError: If the backend_name is not recognized.
    """
    if backend_name == "native":
        return NativeBackend(**kwargs)
    elif backend_name == "langsmith":
        return LangSmithBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown observability backend: '{backend_name}'. "
            f"Supported options: 'native', 'langsmith'."
        )


__all__ = ["get_backend", "ObservabilityBackend", "NativeBackend", "LangSmithBackend"]

