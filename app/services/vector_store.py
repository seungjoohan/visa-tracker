"""
FAISS Vector Store for the RAG knowledge base.

=== HOW VECTOR SEARCH WORKS ===

1. EMBEDDING: We convert text into dense vectors (arrays of floats) using a
   neural network (sentence-transformers). Each piece of text becomes a point
   in 384-dimensional space. Texts with similar meaning end up near each other.

2. INDEXING: FAISS stores these vectors in an optimized data structure (index)
   that allows fast nearest-neighbor search. For the scale of ~10K chunks,
   a flat index (brute-force) is fine. For millions of vectors, you'd use
   approximate nearest neighbor (ANN) indexes like IVF or HNSW.

3. SEARCHING: Given a query vector, FAISS returns the k closest vectors
   (and their distances). We map those back to our PolicyChunk objects
   to get the actual text content.

=== WHY FAISS? ===

- Free and runs locally (no API costs, no cloud dependency)
- Fast: even brute-force search over 10K vectors takes <1ms
- Battle-tested: used by Meta for billion-scale search
- Simple: just numpy arrays + a few FAISS calls
- Persistence: save/load index to/from disk files

=== KEY DESIGN DECISIONS ===

- We use IndexFlatIP (Inner Product) with L2-normalized vectors.
  When vectors are normalized to unit length, inner product = cosine similarity.
  This is the standard approach for semantic search.

- We store metadata (chunk text, source info, etc.) separately in a JSON file,
  keyed by position in the FAISS index. FAISS only stores vectors, not metadata.

- We prepend the context_header to text before embedding. This way, the embedding
  captures not just "what the text says" but also "where it lives in the document
  hierarchy." For example, "specialty occupation" by itself is vague, but
  "USCIS Policy Manual > H-1B > Specialty Occupation: specialty occupation..."
  creates a much more specific embedding.
"""

import faiss
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

from app.models.policy import PolicyChunk, PolicyChunkWithScore, KnowledgeBaseStats, SourceType

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for policy document chunks.

    Architecture:
    - FAISS index: stores only the embedding vectors (numpy float32 arrays)
    - Metadata store: JSON file mapping index position → PolicyChunk data
    - Both files saved/loaded together to keep them in sync

    The separation exists because FAISS is a pure vector search engine — it
    doesn't know about metadata. We need a side-car data structure to map
    from "FAISS result at position 42" back to the actual PolicyChunk object.
    """

    def __init__(
        self,
        index_dir: str,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        embedding_dimension: int = 768,
    ):
        """
        Args:
            index_dir: Directory to save/load the FAISS index and metadata.
            embedding_model_name: Sentence-transformer model to use for embeddings.
                We use the same model as the existing SemanticClassifier for consistency.
                all-MiniLM-L6-v2 outputs 384-dimensional vectors.
            embedding_dimension: Must match the model's output dimension.
                all-MiniLM-L6-v2 = 384, all-mpnet-base-v2 = 768.
        """
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model_name
        self.embedding_dimension = embedding_dimension

        # FAISS index — initialized lazily (created on first add or loaded from disk)
        self.index: Optional[faiss.IndexFlatIP] = None

        # Metadata: list of PolicyChunk dicts, ordered to match FAISS index positions
        # chunks_metadata[i] corresponds to the vector at FAISS index position i
        self.chunks_metadata: List[dict] = []

        # Embedding model — loaded lazily to avoid slow startup
        self._model: Optional[SentenceTransformer] = None

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """
        Lazy-load the embedding model.

        Why lazy? Loading a transformer model takes ~2-3 seconds and ~300MB RAM.
        We don't want that cost on every import or when just reading metadata.
        The model is loaded once and cached for the lifetime of this instance.
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to embedding vectors, normalized to unit length.

        Steps:
        1. The sentence-transformer encodes each text into a 384-dim vector
        2. We L2-normalize each vector so its length = 1.0
        3. After normalization, dot product = cosine similarity

        Why normalize? FAISS IndexFlatIP computes dot products. For unnormalized
        vectors, dot product depends on both angle AND magnitude. We only care
        about angle (semantic direction), so we normalize out the magnitude.

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of shape (len(texts), 384), dtype float32, L2-normalized
        """
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # L2 normalize: divide each vector by its length
        # faiss.normalize_L2 modifies the array in-place for efficiency
        faiss.normalize_L2(embeddings)

        return embeddings.astype(np.float32)

    # -------------------------------------------------------------------------
    # Index Management
    # -------------------------------------------------------------------------

    def _ensure_index(self):
        """Create a new empty FAISS index if one doesn't exist yet."""
        if self.index is None:
            # IndexFlatIP = flat (brute-force) index using inner product
            # "Flat" means no approximation — exact nearest neighbor search
            # Fine for up to ~100K vectors; use IndexIVFFlat for millions
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            logger.info(
                f"Created new FAISS index (dimension={self.embedding_dimension})"
            )

    def add_chunks(self, chunks: List[PolicyChunk]) -> int:
        """
        Add policy chunks to the vector store.

        Process:
        1. Prepare text for embedding (prepend context_header to text)
        2. Embed all texts in a batch (efficient GPU/CPU usage)
        3. Add embeddings to FAISS index
        4. Store metadata alongside

        Args:
            chunks: List of PolicyChunk objects to add

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        self._ensure_index()

        # Prepare text for embedding: context_header + text
        # This is a key RAG technique — by including the hierarchical context,
        # the embedding captures WHERE the information lives, not just WHAT it says
        texts_to_embed = [
            f"{chunk.context_header}: {chunk.text}" for chunk in chunks
        ]

        # Batch embed all texts
        embeddings = self._embed_texts(texts_to_embed)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata (serialize Pydantic models to dicts for JSON storage)
        for chunk in chunks:
            self.chunks_metadata.append(chunk.model_dump(mode="json"))

        logger.info(
            f"Added {len(chunks)} chunks to vector store "
            f"(total: {self.index.ntotal})"
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[SourceType] = None,
        score_threshold: float = 0.0,
    ) -> List[PolicyChunkWithScore]:
        """
        Search the vector store for chunks most relevant to a query.

        This is the core RAG retrieval step:
        1. Embed the query using the same model used for indexing
        2. Ask FAISS for the top_k nearest neighbors
        3. Filter by source type and score threshold
        4. Return PolicyChunk objects with similarity scores

        Args:
            query: Natural language query (e.g., "What are H-1B specialty occupation requirements?")
            top_k: Number of results to return. Typical values:
                   - 3-5 for focused questions
                   - 5-10 for broad topics
                   More results = more context for the LLM, but also more noise
            source_filter: Optional filter to only search specific document types
            score_threshold: Minimum similarity score (0.0 to 1.0) to include

        Returns:
            List of PolicyChunkWithScore, sorted by score (highest first)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty, returning no results")
            return []

        # Embed the query with the same model
        query_embedding = self._embed_texts([query])

        # Search FAISS — returns (distances, indices) arrays
        # distances[0] = array of similarity scores for the query
        # indices[0] = array of positions in our metadata list
        # We fetch more than top_k in case some get filtered out
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            score = float(score)
            if score < score_threshold:
                continue

            chunk_data = self.chunks_metadata[idx]

            # Apply source filter if specified
            if source_filter and chunk_data["source_type"] != source_filter.value:
                continue

            chunk = PolicyChunk(**chunk_data)
            results.append(PolicyChunkWithScore(chunk=chunk, score=score))

            if len(results) >= top_k:
                break

        return results

    # -------------------------------------------------------------------------
    # Persistence (Save / Load)
    # -------------------------------------------------------------------------

    def save(self):
        """
        Save the FAISS index and metadata to disk.

        Two files are saved:
        - faiss.index: Binary file containing the FAISS index (vectors)
        - metadata.json: JSON file containing chunk metadata

        These MUST be kept in sync — if you delete one, delete both.
        The position in metadata.json[i] corresponds to FAISS vector at position i.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)

        index_path = self.index_dir / "faiss.index"
        metadata_path = self.index_dir / "metadata.json"

        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path} ({self.index.ntotal} vectors)")

        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "chunks": self.chunks_metadata,
                    "embedding_model": self.embedding_model_name,
                    "embedding_dimension": self.embedding_dimension,
                    "last_updated": datetime.utcnow().isoformat(),
                },
                f,
                indent=2,
            )
        logger.info(f"Saved metadata to {metadata_path} ({len(self.chunks_metadata)} chunks)")

    def load(self) -> bool:
        """
        Load a previously saved FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False if files don't exist
        """
        index_path = self.index_dir / "faiss.index"
        metadata_path = self.index_dir / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            logger.info(f"No existing index found at {self.index_dir}")
            return False

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r") as f:
            data = json.load(f)

        self.chunks_metadata = data["chunks"]

        # Validate consistency: FAISS vector count should match metadata count
        if self.index.ntotal != len(self.chunks_metadata):
            raise ValueError(
                f"Index/metadata mismatch: FAISS has {self.index.ntotal} vectors "
                f"but metadata has {len(self.chunks_metadata)} entries. "
                f"Delete both files and re-ingest."
            )

        logger.info(
            f"Loaded vector store from {self.index_dir}: "
            f"{self.index.ntotal} vectors, {len(self.chunks_metadata)} chunks"
        )
        return True

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_stats(self) -> KnowledgeBaseStats:
        """Get statistics about the current knowledge base."""
        chunks_by_source: dict[str, int] = {}
        for chunk in self.chunks_metadata:
            source = chunk.get("source_type", "unknown")
            chunks_by_source[source] = chunks_by_source.get(source, 0) + 1

        return KnowledgeBaseStats(
            total_chunks=len(self.chunks_metadata),
            chunks_by_source=chunks_by_source,
            embedding_model=self.embedding_model_name,
            embedding_dimension=self.embedding_dimension,
        )

    def clear(self):
        """Remove all data from the vector store (in-memory only, call save() to persist)."""
        self.index = None
        self.chunks_metadata = []
        logger.info("Cleared vector store")

    def delete_chunks_by_source(self, source_type: SourceType) -> int:
        """
        Delete all chunks from a specific source.

        FAISS doesn't natively support deletion from a flat index, so we rebuild
        the entire index without the deleted chunks. This is fine for our scale
        but would be expensive for millions of vectors.

        Why would we need this? When we re-ingest updated documents from a source,
        we first delete the old chunks, then add the new ones.

        Args:
            source_type: The source type to delete

        Returns:
            Number of chunks deleted
        """
        if not self.chunks_metadata:
            return 0

        # Find indices to keep
        keep_indices = []
        delete_count = 0
        for i, chunk in enumerate(self.chunks_metadata):
            if chunk.get("source_type") == source_type.value:
                delete_count += 1
            else:
                keep_indices.append(i)

        if delete_count == 0:
            return 0

        # Rebuild: extract kept vectors from old index, build new index
        if keep_indices and self.index is not None:
            # Reconstruct vectors for kept indices
            kept_vectors = np.array(
                [self.index.reconstruct(i) for i in keep_indices],
                dtype=np.float32,
            )
            new_metadata = [self.chunks_metadata[i] for i in keep_indices]

            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.index.add(kept_vectors)
            self.chunks_metadata = new_metadata
        else:
            self.clear()

        logger.info(
            f"Deleted {delete_count} chunks from source '{source_type.value}' "
            f"(remaining: {len(self.chunks_metadata)})"
        )
        return delete_count
