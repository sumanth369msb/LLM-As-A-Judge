# ==============================================================================
# Native Observability Backend — SQLite (Local, Zero External Dependencies)
#
# Stores shadow evaluation results in a local SQLite database.
# Generates terminal-based drift reports by querying the database directly.
#
# Usage:
#   Set observability_backend: "native" in config.yaml
#   Results are saved to: data/shadow_evals.db (configurable via db_path)
# ==============================================================================

import os
import sqlite3
from datetime import datetime, timedelta

from .base import ObservabilityBackend


class NativeBackend(ObservabilityBackend):
    """
    SQLite-based observability backend for local development and testing.
    Automatically creates the database and table on first use.
    """

    def __init__(self, db_path: str = "data/shadow_evals.db", drift_threshold: float = 2.0, **kwargs):
        self.db_path = db_path
        self.drift_threshold = drift_threshold

        # Ensure the directory for the database exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize the database schema
        self._init_db()

    def _init_db(self):
        """Creates the shadow_evaluations table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT NOT NULL,
                judge_model TEXT NOT NULL,
                faithfulness_score REAL NOT NULL,
                faithfulness_justification TEXT,
                relevancy_score REAL NOT NULL,
                relevancy_justification TEXT,
                latency_ms REAL,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

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
        """Inserts a single evaluation result into the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO shadow_evaluations
                (incident_id, judge_model, faithfulness_score, faithfulness_justification,
                 relevancy_score, relevancy_justification, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                incident_id,
                judge_model,
                faithfulness_score,
                faithfulness_justification,
                relevancy_score,
                relevancy_justification,
                latency_ms,
            ),
        )
        conn.commit()
        conn.close()

    def generate_report(self, days: int = 7) -> None:
        """
        Queries the SQLite database and prints a terminal-based drift report
        covering the last N days of shadow evaluations.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Total evaluations in the window
        cursor.execute(
            "SELECT COUNT(*) FROM shadow_evaluations WHERE evaluated_at >= ?",
            (cutoff,),
        )
        total = cursor.fetchone()[0]

        if total == 0:
            print(f"\n[DRIFT REPORT] No evaluations found in the last {days} days.")
            conn.close()
            return

        # Average scores
        cursor.execute(
            """
            SELECT
                ROUND(AVG(faithfulness_score), 2),
                ROUND(AVG(relevancy_score), 2),
                ROUND(AVG(latency_ms), 1)
            FROM shadow_evaluations
            WHERE evaluated_at >= ?
            """,
            (cutoff,),
        )
        avg_faith, avg_relev, avg_latency = cursor.fetchone()

        # Count drift alerts (scores at or below threshold)
        cursor.execute(
            """
            SELECT COUNT(*) FROM shadow_evaluations
            WHERE evaluated_at >= ?
              AND (faithfulness_score <= ? OR relevancy_score <= ?)
            """,
            (cutoff, self.drift_threshold, self.drift_threshold),
        )
        drift_count = cursor.fetchone()[0]

        conn.close()

        # Print the report
        drift_pct = (drift_count / total) * 100 if total > 0 else 0.0

        print(f"\n{'=' * 60}")
        print(f"  DRIFT MONITORING REPORT — Last {days} Days")
        print(f"{'=' * 60}")
        print(f"  Total Evaluations:       {total}")
        print(f"  Avg Faithfulness:        {avg_faith}/5")
        print(f"  Avg Relevancy:           {avg_relev}/5")
        print(f"  Avg Latency:             {avg_latency}ms")
        print(f"  Drift Alerts:            {drift_count} ({drift_pct:.1f}%)")
        print(f"  Drift Threshold:         <= {self.drift_threshold}")
        print(f"  Database:                {self.db_path}")
        print(f"{'=' * 60}")

        if drift_pct > 20:
            print("  [WARNING] Drift rate exceeds 20%. Investigate model quality.")
        elif drift_pct > 0:
            print("  [INFO] Some drift detected. Monitor closely.")
        else:
            print("  [OK] No drift detected. Model quality is stable.")
        print()
