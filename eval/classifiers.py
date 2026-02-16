"""
Classifier adapters for evaluation.

=== THE ADAPTER PATTERN ===

Each classifier in our system has a different interface:
  - KeywordOnly: sync keyword matching, no models loaded
  - Semantic: needs sentence-transformer model (slow to load)
  - LLM: async API call, costs money, returns structured JSON
  - LLM+RAG: async API + vector store search, costs more

For evaluation, we need a UNIFORM interface so the eval CLI can loop over
any classifier without caring how it works internally. That's what these
adapters provide: each takes a LabeledArticle and returns a ClassificationResult.

=== ADAPTER → NEWSANALYZER MAPPING ===

Under the hood, each adapter creates a NewsAnalyzer with different components
enabled/disabled. The NewsAnalyzer already has the full fallback chain:
    LLM + RAG → LLM only → Semantic + Keyword → Keyword only

By controlling which components are injected, we isolate each method:

    Adapter           | use_semantic | llm_provider | retriever
    ──────────────────┼──────────────┼──────────────┼──────────
    KeywordOnly       | False        | None         | None
    Semantic          | True         | None         | None
    LLMOnly           | True         | ClaudeProvider| None      ← no RAG context
    LLMWithRAG        | True         | ClaudeProvider| Retriever ← full pipeline
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from eval.models import LabeledArticle, ClassificationResult
from app.models.news import NewsArticle, NewsSource, ImportanceLevel
from app.services.news_analyzer import NewsAnalyzer
from app.services.llm_provider import ClaudeProvider
from app.services.retriever import Retriever
from app.services.llm_usage_logger import LLMUsageLogger
from app.services.vector_store import VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)


def labeled_to_news_article(article: LabeledArticle) -> NewsArticle:
    """
    Convert a LabeledArticle back to a NewsArticle for the analyzer.

    LabeledArticle uses plain strings for serialization (dates, enums).
    NewsArticle uses proper types (datetime, Enum). This bridges the gap.
    """
    # Map source string back to enum
    source_map = {s.value: s for s in NewsSource}
    source = source_map.get(article.source, NewsSource.NEWS_API)

    return NewsArticle(
        title=article.title,
        description=article.description,
        content=article.content,
        url=article.url,
        published_at=datetime.fromisoformat(article.published_at),
        source=source,
        keywords=article.keywords,
        importance_level=None,
        relevance_score=None,
        summary=None,
        analysis_result=None,
    )


class EvalClassifier(ABC):
    """
    Abstract base class for evaluation classifiers.

    Every adapter must implement:
      - name: human-readable identifier
      - requires_llm: whether it costs money (for the --confirm-cost check)
      - classify(): run classification on one article
    """

    name: str
    requires_llm: bool = False

    @abstractmethod
    async def classify(self, article: LabeledArticle) -> ClassificationResult:
        """Classify one article. Returns prediction + timing + cost."""
        ...

    async def setup(self):
        """Optional setup (e.g., loading models). Called once before eval loop."""
        pass

    async def teardown(self):
        """Optional cleanup. Called after eval loop."""
        pass


class KeywordOnlyClassifier(EvalClassifier):
    """
    Keyword-only classification — the simplest baseline.

    No models loaded, no API calls. Just regex matching against
    urgent_keywords and important_keywords lists from config.
    """

    name = "keyword"
    requires_llm = False

    def __init__(self):
        self._analyzer: Optional[NewsAnalyzer] = None

    async def setup(self):
        self._analyzer = NewsAnalyzer(
            use_semantic=False,
            llm_provider=None,
            retriever=None,
            usage_logger=None,
        )

    async def classify(self, article: LabeledArticle) -> ClassificationResult:
        news_article = labeled_to_news_article(article)

        start = time.monotonic()
        level = await self._analyzer.classify_importance(news_article)
        elapsed = (time.monotonic() - start) * 1000

        return ClassificationResult(
            predicted_label=level.value,
            latency_ms=elapsed,
            cost_usd=0.0,
        )


class SemanticClassifier(EvalClassifier):
    """
    Semantic + keyword classification.

    Loads the sentence-transformer model (all-MiniLM-L6-v2) and uses
    cosine similarity to pre-computed example embeddings. Falls back
    to keyword logic when confidence is low.
    """

    name = "semantic"
    requires_llm = False

    def __init__(self):
        self._analyzer: Optional[NewsAnalyzer] = None

    async def setup(self):
        # use_semantic=True triggers model loading in NewsAnalyzer.__init__
        self._analyzer = NewsAnalyzer(
            use_semantic=True,
            llm_provider=None,
            retriever=None,
            usage_logger=None,
        )

    async def classify(self, article: LabeledArticle) -> ClassificationResult:
        news_article = labeled_to_news_article(article)

        start = time.monotonic()
        level = await self._analyzer.classify_importance(news_article)
        elapsed = (time.monotonic() - start) * 1000

        return ClassificationResult(
            predicted_label=level.value,
            latency_ms=elapsed,
            cost_usd=0.0,
        )


class LLMOnlyClassifier(EvalClassifier):
    """
    LLM classification WITHOUT RAG context.

    Sends the article to Claude Haiku but does NOT retrieve policy chunks
    from the vector store. This tests how well the LLM classifies using
    only its training knowledge (no grounding in official policy docs).

    Compare with LLMWithRAGClassifier to measure the VALUE of RAG.
    """

    name = "llm"
    requires_llm = True

    def __init__(self):
        self._analyzer: Optional[NewsAnalyzer] = None
        self._usage_logger: Optional[LLMUsageLogger] = None

    async def setup(self):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for LLM classifier")

        provider = ClaudeProvider(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
        )
        self._usage_logger = LLMUsageLogger(settings.logs_dir)

        # retriever=None → no RAG context (the key difference from LLMWithRAG)
        self._analyzer = NewsAnalyzer(
            use_semantic=True,
            llm_provider=provider,
            retriever=None,
            usage_logger=self._usage_logger,
        )

    async def classify(self, article: LabeledArticle) -> ClassificationResult:
        news_article = labeled_to_news_article(article)

        start = time.monotonic()

        # Try LLM analysis first
        analysis = await self._analyzer.analyze_article(news_article)
        if analysis:
            label = analysis.importance
            confidence = analysis.confidence
        else:
            # Fallback if LLM fails
            level = await self._analyzer.classify_importance(news_article)
            label = level.value
            confidence = None

        elapsed = (time.monotonic() - start) * 1000

        # Estimate cost from the last LLM call
        cost = 0.0
        if analysis and self._usage_logger and self._usage_logger._last_cost:
            cost = self._usage_logger._last_cost

        return ClassificationResult(
            predicted_label=label,
            latency_ms=elapsed,
            cost_usd=cost,
            confidence=confidence,
        )


class LLMWithRAGClassifier(EvalClassifier):
    """
    Full LLM + RAG classification — the production pipeline.

    Retrieves relevant policy chunks from the FAISS vector store,
    includes them in the prompt, then calls Claude Haiku. This is
    the most expensive but (hopefully) most accurate method.
    """

    name = "llm_rag"
    requires_llm = True

    def __init__(self):
        self._analyzer: Optional[NewsAnalyzer] = None
        self._usage_logger: Optional[LLMUsageLogger] = None

    async def setup(self):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for LLM+RAG classifier")

        provider = ClaudeProvider(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
        )

        # Load the FAISS vector store
        store = VectorStore(
            index_dir=str(settings.index_dir),
            embedding_model_name=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
        )
        store.load()

        retriever = Retriever(store, top_k=settings.retrieval_top_k)
        self._usage_logger = LLMUsageLogger(settings.logs_dir)

        self._analyzer = NewsAnalyzer(
            use_semantic=True,
            llm_provider=provider,
            retriever=retriever,
            usage_logger=self._usage_logger,
        )

    async def classify(self, article: LabeledArticle) -> ClassificationResult:
        news_article = labeled_to_news_article(article)

        start = time.monotonic()

        analysis = await self._analyzer.analyze_article(news_article)
        if analysis:
            label = analysis.importance
            confidence = analysis.confidence
        else:
            level = await self._analyzer.classify_importance(news_article)
            label = level.value
            confidence = None

        elapsed = (time.monotonic() - start) * 1000

        cost = 0.0
        if analysis and self._usage_logger and self._usage_logger._last_cost:
            cost = self._usage_logger._last_cost

        return ClassificationResult(
            predicted_label=label,
            latency_ms=elapsed,
            cost_usd=cost,
            confidence=confidence,
        )


# Registry: map classifier name to class
CLASSIFIERS = {
    "keyword": KeywordOnlyClassifier,
    "semantic": SemanticClassifier,
    "llm": LLMOnlyClassifier,
    "llm_rag": LLMWithRAGClassifier,
}
