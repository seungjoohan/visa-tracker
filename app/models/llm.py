"""
Data models for the LLM integration layer.

Three models, three concerns:

1. LLMResponse: What comes back from the LLM API (raw text + metadata).
   Provider-agnostic — Claude, GPT, and Gemini all produce this same shape.

2. AnalysisResult: The parsed, domain-specific output after we extract JSON
   from the LLM's raw text. This is what downstream code (email, API) consumes.

3. LLMUsageRecord: Cost/latency tracking for every LLM call.

Why separate LLMResponse from AnalysisResult? The LLM provider is a generic
text-generation interface — it shouldn't know about visa types or policy refs.
The *caller* (news_analyzer) parses the raw text into AnalysisResult. This means
we can reuse the same provider for classification, summarization, and Q&A,
each with their own output schemas.
"""

from pydantic import BaseModel
from datetime import datetime, date
from typing import Optional, List


class LLMResponse(BaseModel):
    """
    Raw response from any LLM provider.

    Every LLM API returns: generated text, token counts, and timing.
    We compute cost from token counts using provider-specific pricing.
    """
    text: str
    model: str                  # e.g., "claude-3-5-haiku-20241022"
    provider: str               # e.g., "anthropic", "openai"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime


class AnalysisResult(BaseModel):
    """
    Structured analysis of a news article, parsed from LLM JSON output.

    The LLM prompt includes this schema as the expected output format.
    Pydantic validates the parsed JSON — if fields are missing or wrong type,
    we catch it and fall back to keyword+semantic classification.
    """
    importance: str                     # "needs_attention" | "good_to_know" | "no_attention_required"
    affected_visa_types: List[str]      # e.g., ["H-1B", "F-1"]
    action_required: bool
    deadline: Optional[date] = None
    impact_summary: str                 # 2-3 sentence practical summary
    relevant_policy_refs: List[str]     # e.g., ["8 CFR 214.2(h)"]
    confidence: float                   # 0.0 to 1.0


class LLMUsageRecord(BaseModel):
    """
    Per-call cost/latency record, written to logs/llm_usage.jsonl.

    JSONL (one JSON object per line) is easy to append, parse, and load
    into pandas for analysis. No database dependency needed.
    """
    timestamp: datetime
    provider: str
    model: str
    operation: str              # "classify", "summarize", "qa"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    error: Optional[str] = None
