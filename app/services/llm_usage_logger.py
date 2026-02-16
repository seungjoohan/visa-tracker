"""
LLM usage/cost logger.

Appends one JSON line per LLM call to logs/llm_usage.jsonl.

JSONL (JSON Lines) format: one complete JSON object per line.
    {"timestamp": "2025-01-15T...", "provider": "anthropic", "cost_usd": 0.000032, ...}
    {"timestamp": "2025-01-15T...", "provider": "anthropic", "cost_usd": 0.000028, ...}

Why JSONL instead of a database?
- Append-only: no locks, no corruption risk, works with concurrent processes
- Human-readable: cat/grep/jq for quick inspection
- Easy to load into pandas: pd.read_json("llm_usage.jsonl", lines=True)
- No dependencies: just file I/O
- Sufficient for ~50 records/day
"""

import logging
from pathlib import Path
from typing import Optional

from app.models.llm import LLMResponse, LLMUsageRecord

logger = logging.getLogger(__name__)


class LLMUsageLogger:

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_file = self.log_dir / "llm_usage.jsonl"
        # Track last call's cost so eval classifiers can report per-article cost
        self._last_cost: float = 0.0

    def log_call(
        self,
        response: LLMResponse,
        operation: str,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """
        Append a usage record for one LLM call.

        Args:
            response: The LLMResponse from the provider
            operation: What triggered this call ("classify", "summarize", "qa")
            success: Whether we successfully parsed the response
            error: Error message if success=False
        """
        self._last_cost = response.cost_usd

        record = LLMUsageRecord(
            timestamp=response.timestamp,
            provider=response.provider,
            model=response.model,
            operation=operation,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            success=success,
            error=error,
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(record.model_dump_json() + "\n")
