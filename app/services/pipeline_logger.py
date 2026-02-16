"""
Pipeline stage logger — timing and metrics for each daily pipeline step.

=== PURPOSE ===

The existing LLMUsageLogger tracks per-LLM-call metrics (tokens, cost).
This logger tracks per-PIPELINE-STAGE metrics (latency, article counts).

Together they give you a complete picture:
  - PipelineStageLogger: "The collect stage took 3.2s and found 47 articles"
  - LLMUsageLogger: "The LLM classified 12 articles at $0.0024 total"

=== HOW IT WORKS ===

Uses a context manager to wrap each pipeline stage:

    with logger.log_stage("collect", {"keywords": 8}):
        articles = await collector.collect_news(keywords)
        logger.update_stage({"article_count": len(articles)})

This measures the wall-clock time of the stage and writes a JSONL record
to logs/pipeline_metrics.jsonl.

=== WHY NOT JUST USE LOGGING? ===

Structured JSONL lets you:
  - Load into pandas for analysis: pd.read_json(path, lines=True)
  - Query with jq: cat pipeline_metrics.jsonl | jq 'select(.stage=="analyze")'
  - Track trends over time (are we getting slower?)
  - Include in evaluation reports

Regular logging (logger.info) is for humans reading logs in real-time.
JSONL is for programmatic analysis after the fact.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class PipelineStageLogger:

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_file = self.log_dir / "pipeline_metrics.jsonl"
        self._current_stage: Optional[dict] = None

    @contextmanager
    def log_stage(self, stage_name: str, metadata: dict = None):
        """
        Context manager that times a pipeline stage and logs it.

        Args:
            stage_name: Identifier for the stage (collect, filter, analyze, email)
            metadata: Optional dict of extra fields to include in the log record

        Usage:
            with logger.log_stage("filter", {"threshold": 0.5}):
                results = await analyzer.filter_relevant_articles(articles)
                logger.update_stage({"kept": len(results), "total": len(all)})
        """
        self._current_stage = {
            "timestamp": datetime.utcnow().isoformat(),
            "stage": stage_name,
            **(metadata or {}),
        }

        start = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            self._current_stage["latency_ms"] = round(elapsed_ms, 2)

            # Write to JSONL
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(self._current_stage) + "\n")

            logger.info(
                f"Pipeline stage '{stage_name}' completed in {elapsed_ms:.0f}ms"
            )
            self._current_stage = None

    def update_stage(self, metadata: dict):
        """
        Add extra fields to the current stage's log record.

        Call this inside the log_stage context manager to add data
        that's only available after the stage runs (e.g., article counts).
        """
        if self._current_stage:
            self._current_stage.update(metadata)
