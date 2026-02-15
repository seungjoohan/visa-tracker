"""
LLM Provider interface and implementations.

=== THE STRATEGY PATTERN ===

We define an abstract base class (LLMProvider) with a single method: generate().
Concrete classes (ClaudeProvider, OpenAIProvider, etc.) implement the actual API call.
The caller (news_analyzer) only knows about the ABC — it doesn't care which provider
is behind it. This lets us swap providers without changing any analysis code.

    LLMProvider (ABC)
        │
        ├── ClaudeProvider      ← Phase 1 (now)
        ├── OpenAIProvider      ← Phase 5
        ├── GeminiProvider      ← Phase 5
        └── LocalProvider       ← Phase 5 (ollama)

=== WHY START WITH CLAUDE HAIKU? ===

- Cheapest Claude model: ~$0.80/M input, $4.00/M output tokens
- Fast: ~100-200ms for short responses
- Excellent at structured JSON output
- For 50 articles/day × ~800 token prompts: ~$0.03/day
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from app.models.llm import LLMResponse

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Any provider must implement generate() which takes a prompt and returns
    an LLMResponse containing the raw text, token counts, cost, and latency.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt: The user message / main prompt
            system: System message (role/behavior instructions)
            max_tokens: Maximum tokens in the response
            temperature: 0.0 = deterministic, 1.0 = creative.
                Use 0.0 for classification (consistent results).
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str: ...

    @abstractmethod
    def get_model_name(self) -> str: ...


class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Uses the anthropic Python SDK's async client. The client is lazy-initialized
    to avoid import/connection overhead when the provider is instantiated but
    not yet used.
    """

    # Pricing per 1M tokens (update when Anthropic changes pricing)
    PRICING = {
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """
        Lazy-initialize the Anthropic async client.

        Why lazy? Importing anthropic and creating a client involves network
        setup (connection pooling). We don't want that cost at DI time —
        only when we actually make the first API call.
        """
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        client = self._get_client()

        # time.monotonic() is better than time.time() for measuring durations —
        # it's not affected by system clock adjustments
        start = time.monotonic()

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)

        elapsed_ms = (time.monotonic() - start) * 1000

        # Calculate cost from token counts
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        pricing = self.PRICING.get(self.model, {"input": 1.0, "output": 5.0})
        cost = (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        ) / 1_000_000

        result = LLMResponse(
            text=response.content[0].text,
            model=self.model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=elapsed_ms,
            timestamp=datetime.utcnow(),
        )

        logger.info(
            f"LLM call: model={self.model}, "
            f"tokens={input_tokens}+{output_tokens}, "
            f"cost=${cost:.6f}, latency={elapsed_ms:.0f}ms"
        )

        return result

    def get_provider_name(self) -> str:
        return "anthropic"

    def get_model_name(self) -> str:
        return self.model
