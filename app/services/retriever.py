"""
RAG Retrieval layer.

=== WHERE THIS FITS IN THE RAG PIPELINE ===

    User Query / Article Text
            │
            ▼
      ┌─────────────┐
      │  Retriever   │  ← THIS MODULE
      │              │
      │  1. Embed    │  (delegates to VectorStore)
      │  2. Search   │  (delegates to VectorStore)
      │  3. Format   │  (own logic)
      └──────┬───────┘
             │
             ▼
      Formatted context string
      ready to paste into LLM prompt

The Retriever sits between the VectorStore (which does raw vector search)
and the LLM prompt (which needs human-readable context). Its job:

1. Take a query (article text or user question)
2. Search the vector store for relevant policy chunks
3. Format results into a numbered, cited context string

Why not do this in the analyzer directly? Two reasons:
- The Q&A chatbot (Phase 4) needs the same retrieval + formatting logic
- Formatting for prompts has its own complexity (truncation, citations,
  deduplication) that would clutter the analyzer
"""

import logging
from typing import List, Optional

from app.services.vector_store import VectorStore
from app.models.policy import PolicyChunkWithScore

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[PolicyChunkWithScore]:
        """Search vector store and return raw results."""
        k = top_k or self.top_k
        return self.vector_store.search(query=query, top_k=k)

    def retrieve_and_format(
        self, query: str, top_k: Optional[int] = None
    ) -> str:
        """
        Retrieve chunks and format them as a prompt-ready context string.

        Output format:
            [1] (relevance: 0.82) 8 CFR 214.2(h) | CFR Title 8 > H-1B > ...
            <chunk text>

            ---

            [2] (relevance: 0.71) ...
            <chunk text>

        Returns empty string if no results found — the LLM can still attempt
        classification without policy context, just with lower confidence.
        """
        results = self.retrieve(query, top_k)

        if not results:
            logger.info(f"No policy context found for query: {query[:80]}...")
            return ""

        parts = []
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            ref = chunk.cfr_reference or "No CFR ref"
            header = chunk.context_header
            parts.append(
                f"[{i}] (relevance: {result.score:.2f}) {ref} | {header}\n"
                f"{chunk.text}"
            )

        context = "\n\n---\n\n".join(parts)
        logger.info(
            f"Retrieved {len(results)} chunks "
            f"(top score: {results[0].score:.2f})"
        )
        return context
