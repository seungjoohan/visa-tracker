"""
Policy document chunk models for the RAG knowledge base.

In a RAG system, we don't store entire documents — we break them into "chunks"
(typically 500-1000 tokens each). Each chunk becomes a single entry in the
vector store, with its own embedding vector for similarity search.

Why these fields?
- id: Unique identifier so we can update/delete specific chunks
- text: The actual content that gets embedded and searched against
- context_header: A breadcrumb trail (e.g., "USCIS Policy Manual > H-1B > Specialty Occupation")
  that provides hierarchical context. This gets prepended to the text before embedding
  so the embedding captures both the content AND where it lives in the document hierarchy.
- source_type: Which document collection this came from (for filtering retrieval)
- source_url: Link back to the original source (for citations in LLM responses)
- cfr_reference: Legal citation (e.g., "8 CFR 214.2(h)(4)(ii)") for credibility
- effective_date / last_updated: Temporal metadata so we can prefer newer documents
  and warn users about potentially outdated information
"""

from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Optional, List


class SourceType(Enum):
    """
    Each source type represents a different collection of immigration documents.
    We track this so that during retrieval we can optionally filter by source
    (e.g., "only search official regulations, not news articles").
    """
    USCIS_POLICY_MANUAL = "uscis_policy_manual"
    CFR_TITLE_8 = "cfr_title_8"
    FEDERAL_REGISTER = "federal_register"
    USCIS_MEMO = "uscis_memo"
    DOS_FAM = "dos_fam"  # Department of State Foreign Affairs Manual
    AAO_DECISION = "aao_decision"  # Administrative Appeals Office


class PolicyChunk(BaseModel):
    """
    A single chunk of a policy document, ready to be embedded and stored.

    This is the fundamental unit in our vector store. When a user asks a question
    or when we analyze a news article, we embed the query and find the PolicyChunks
    whose embeddings are most similar (closest in vector space).
    """
    id: str  # e.g., "uscis-pm-vol2-partH-ch5-001"
    text: str  # The actual chunk content (500-1000 tokens)
    context_header: str  # Hierarchical breadcrumb for context
    source_type: SourceType
    source_url: Optional[str] = None
    cfr_reference: Optional[str] = None  # Legal citation if applicable
    effective_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None


class PolicyChunkWithScore(BaseModel):
    """
    A PolicyChunk paired with a similarity score from vector search.

    When FAISS returns results, it gives us (index, distance) pairs.
    We convert the distance to a similarity score (0.0 to 1.0) and
    attach it to the chunk so downstream code can filter by relevance
    or show confidence to the user.
    """
    chunk: PolicyChunk
    score: float  # Cosine similarity score (higher = more relevant)


class KnowledgeBaseStats(BaseModel):
    """
    Metadata about the current state of the knowledge base.
    Useful for monitoring and debugging — "how many chunks do we have?",
    "when was the last ingestion?", "which sources are represented?"
    """
    total_chunks: int
    chunks_by_source: dict[str, int]
    last_updated: Optional[datetime] = None
    embedding_model: str
    embedding_dimension: int
