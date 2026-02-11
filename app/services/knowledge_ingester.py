"""
Knowledge base ingestion pipeline.

=== WHAT THIS DOES ===

This module handles the "offline" part of RAG: taking raw policy documents,
breaking them into chunks, and storing them in the vector store. This runs
periodically (or on-demand), NOT at query time.

=== THE INGESTION PIPELINE ===

    Raw Document (e.g., USCIS Policy Manual webpage)
         │
         ▼
    1. FETCH: Download the raw text from the source
         │
         ▼
    2. CLEAN: Remove HTML, normalize whitespace, extract structure
         │
         ▼
    3. CHUNK: Split into 500-1000 token pieces with overlap
         │
         ▼
    4. ENRICH: Attach metadata (source, section, date, URL)
         │
         ▼
    5. STORE: Embed + add to FAISS vector store

=== CHUNKING STRATEGY ===

We use a hybrid approach:
- First, try to split on natural section boundaries (headings, paragraph breaks)
- If a section is too large, fall back to token-based splitting with overlap
- Preserve the section hierarchy as a "context_header" breadcrumb

Why not just split every N tokens? Because splitting mid-sentence or mid-paragraph
destroys meaning. A chunk that starts with "...the applicant must also demonstrate"
is useless without knowing what came before. Section-aware chunking preserves
complete thoughts.

=== WHY OVERLAP? ===

Even with section-aware chunking, important information can span chunk boundaries.
With 100-token overlap, the last 100 tokens of chunk N repeat at the start of
chunk N+1. This means a concept that falls at a boundary is fully captured in
at least one chunk. The tradeoff: ~10-15% more storage and slightly more
redundancy in search results.
"""

import re
import logging
from typing import List, Optional
from datetime import datetime

from app.models.policy import PolicyChunk, SourceType
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Chunking Utilities
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.

    Why estimate instead of using a real tokenizer? Speed and simplicity.
    Real tokenizers (tiktoken, sentencepiece) add dependencies and are slow
    for bulk processing. The ~4 chars/token heuristic is accurate within ~10%
    for English text, which is good enough for chunking decisions.

    For production RAG systems with strict token budgets, use tiktoken.
    """
    return len(text) // 4  # Rough estimate: ~4 characters per token


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks of approximately chunk_size tokens.

    Algorithm:
    1. Split text into paragraphs (double newline boundaries)
    2. Accumulate paragraphs into a chunk until we hit chunk_size
    3. When full, save the chunk and start a new one
    4. The new chunk starts with the last `chunk_overlap` tokens from the previous chunk

    This preserves paragraph boundaries (better than splitting mid-sentence)
    while maintaining consistent chunk sizes.

    Args:
        text: The text to chunk
        chunk_size: Target size in estimated tokens (default 800)
        chunk_overlap: Overlap between consecutive chunks (default 100)

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    # If text is small enough, return as single chunk
    if estimate_tokens(text) <= chunk_size:
        return [text.strip()]

    # Split on paragraph boundaries (double newlines)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk_parts = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If a single paragraph exceeds chunk_size, split it by sentences
        if para_tokens > chunk_size:
            # First, flush current chunk if non-empty
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_tokens = 0

            # Split the long paragraph by sentences
            sentence_chunks = _chunk_by_sentences(para, chunk_size, chunk_overlap)
            chunks.extend(sentence_chunks)
            continue

        # Would adding this paragraph exceed the chunk size?
        if current_tokens + para_tokens > chunk_size and current_chunk_parts:
            # Save current chunk
            chunks.append("\n\n".join(current_chunk_parts))

            # Start new chunk with overlap from the end of the previous chunk
            overlap_text = _get_overlap_text(
                current_chunk_parts, chunk_overlap
            )
            if overlap_text:
                current_chunk_parts = [overlap_text]
                current_tokens = estimate_tokens(overlap_text)
            else:
                current_chunk_parts = []
                current_tokens = 0

        current_chunk_parts.append(para)
        current_tokens += para_tokens

    # Don't forget the last chunk
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    return chunks


def _chunk_by_sentences(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """
    Fallback chunking for very long paragraphs: split by sentence boundaries.

    Uses a simple regex to find sentence endings (period/question/exclamation
    followed by space or newline). Not perfect for abbreviations ("U.S.C.I.S.")
    but good enough for legal text.
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_parts = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)

        if current_tokens + sent_tokens > chunk_size and current_parts:
            chunks.append(" ".join(current_parts))
            # Simple overlap: keep last few sentences
            overlap_tokens = 0
            overlap_parts = []
            for part in reversed(current_parts):
                overlap_tokens += estimate_tokens(part)
                if overlap_tokens > chunk_overlap:
                    break
                overlap_parts.insert(0, part)
            current_parts = overlap_parts
            current_tokens = sum(estimate_tokens(p) for p in current_parts)

        current_parts.append(sentence)
        current_tokens += sent_tokens

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _get_overlap_text(parts: List[str], overlap_tokens: int) -> str:
    """
    Get the last `overlap_tokens` worth of text from a list of parts.
    Used to create overlap between consecutive chunks.
    """
    overlap_parts = []
    tokens_so_far = 0

    for part in reversed(parts):
        part_tokens = estimate_tokens(part)
        tokens_so_far += part_tokens
        overlap_parts.insert(0, part)
        if tokens_so_far >= overlap_tokens:
            break

    return "\n\n".join(overlap_parts) if overlap_parts else ""


# ---------------------------------------------------------------------------
# Knowledge Ingester
# ---------------------------------------------------------------------------

class KnowledgeIngester:
    """
    Orchestrates the ingestion of policy documents into the vector store.

    This class is the main entry point for adding documents. It handles:
    - Chunking documents with proper metadata
    - Deduplication (via delete-then-reinsert for a given source)
    - Batch insertion into the vector store
    - Persistence (saving the updated index to disk)

    Usage:
        store = VectorStore(index_dir="knowledge_base/index")
        ingester = KnowledgeIngester(store)

        # Ingest raw sections from USCIS policy manual
        chunks = ingester.ingest_sections(
            sections=parsed_sections,
            source_type=SourceType.USCIS_POLICY_MANUAL,
            base_url="https://www.uscis.gov/policy-manual/...",
        )
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def ingest_sections(
        self,
        sections: List[dict],
        source_type: SourceType,
        base_url: Optional[str] = None,
        replace_existing: bool = True,
    ) -> List[PolicyChunk]:
        """
        Ingest a list of document sections into the vector store.

        Each section should be a dict with:
        - "title": Section title (e.g., "Specialty Occupation Definition")
        - "context_header": Hierarchical breadcrumb
        - "text": The section content
        - "url": (optional) URL to this specific section
        - "cfr_reference": (optional) Legal citation
        - "effective_date": (optional) Date string

        Args:
            sections: List of section dicts
            source_type: Which source these come from
            base_url: Default URL if sections don't have individual URLs
            replace_existing: If True, delete existing chunks from this source
                before adding new ones. This ensures we don't accumulate stale data.

        Returns:
            List of PolicyChunk objects that were created and stored
        """
        if replace_existing:
            deleted = self.vector_store.delete_chunks_by_source(source_type)
            if deleted:
                logger.info(
                    f"Deleted {deleted} existing chunks from {source_type.value}"
                )

        all_chunks = []
        for i, section in enumerate(sections):
            section_chunks = self._process_section(
                section=section,
                source_type=source_type,
                section_index=i,
                base_url=base_url,
            )
            all_chunks.extend(section_chunks)

        if all_chunks:
            self.vector_store.add_chunks(all_chunks)
            self.vector_store.save()

        logger.info(
            f"Ingested {len(all_chunks)} chunks from {len(sections)} sections "
            f"(source: {source_type.value})"
        )

        return all_chunks

    def _process_section(
        self,
        section: dict,
        source_type: SourceType,
        section_index: int,
        base_url: Optional[str],
    ) -> List[PolicyChunk]:
        """
        Process a single section: chunk it and create PolicyChunk objects.

        The ID format is: {source_type}-{section_index}-{chunk_index}
        e.g., "uscis_policy_manual-003-001" = 3rd section, 1st chunk
        This makes IDs deterministic and sortable.
        """
        text = section.get("text", "")
        if not text.strip():
            return []

        context_header = section.get("context_header", section.get("title", ""))
        url = section.get("url", base_url)

        # Parse effective_date if provided as string
        effective_date = None
        if section.get("effective_date"):
            try:
                effective_date = datetime.fromisoformat(section["effective_date"])
            except (ValueError, TypeError):
                pass

        # Chunk the section text
        text_chunks = chunk_text(text)

        # Create PolicyChunk objects for each text chunk
        policy_chunks = []
        for chunk_idx, chunk_text_content in enumerate(text_chunks):
            chunk_id = (
                f"{source_type.value}-{section_index:04d}-{chunk_idx:03d}"
            )

            policy_chunks.append(
                PolicyChunk(
                    id=chunk_id,
                    text=chunk_text_content,
                    context_header=context_header,
                    source_type=source_type,
                    source_url=url,
                    cfr_reference=section.get("cfr_reference"),
                    effective_date=effective_date,
                    last_updated=datetime.utcnow(),
                )
            )

        return policy_chunks

    def ingest_raw_text(
        self,
        text: str,
        title: str,
        source_type: SourceType,
        context_header: Optional[str] = None,
        source_url: Optional[str] = None,
        cfr_reference: Optional[str] = None,
    ) -> List[PolicyChunk]:
        """
        Convenience method: ingest a single raw text document.

        Useful for quick testing or one-off documents. Wraps the text
        in the section format expected by ingest_sections().
        """
        section = {
            "title": title,
            "context_header": context_header or title,
            "text": text,
            "url": source_url,
            "cfr_reference": cfr_reference,
        }
        return self.ingest_sections(
            sections=[section],
            source_type=source_type,
            replace_existing=False,
        )
