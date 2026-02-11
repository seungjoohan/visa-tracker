"""
Compare embedding models for RAG retrieval quality.

=== WHY THIS MATTERS ===

In a RAG system, the embedding model is the "gatekeeper" — it determines which
chunks the LLM sees. A better embedding model means:
- More relevant chunks retrieved → better LLM answers
- Higher score gap between relevant/irrelevant → easier to set thresholds
- Fewer "noise" chunks → less confusion for the LLM

The tradeoff is always quality vs. speed/cost:
- Larger models produce better embeddings but are slower and use more memory
- For offline ingestion (runs once), speed barely matters
- For real-time search, latency matters but FAISS search is fast regardless

=== WHAT WE MEASURE ===

1. Top-1 accuracy: Does the best result match the expected topic?
2. Score magnitude: Higher absolute scores = more confident matches
3. Score gap: Difference between #1 and #2 — larger gap = clearer separation
4. Ingestion speed: How long to embed all chunks
5. Search latency: Time per query

=== USAGE ===

    # Auto-detect dimension for any sentence-transformers model:
    python scripts/compare_embeddings.py all-MiniLM-L6-v2
    python scripts/compare_embeddings.py all-mpnet-base-v2
    python scripts/compare_embeddings.py BAAI/bge-small-en-v1.5
    python scripts/compare_embeddings.py intfloat/e5-small-v2

    # Compare multiple models (runs each in sequence):
    python scripts/compare_embeddings.py all-MiniLM-L6-v2 all-mpnet-base-v2 BAAI/bge-small-en-v1.5

    # Override dimension manually if auto-detect fails:
    python scripts/compare_embeddings.py --dim 384 some-custom-model

    # Show per-query details:
    python scripts/compare_embeddings.py --verbose all-mpnet-base-v2
"""

import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import VectorStore
from app.services.knowledge_ingester import KnowledgeIngester
from app.models.policy import SourceType

# Use fallback content for consistent comparison
from scripts.ingest_uscis import _get_fallback_h1b_content

TEST_QUERIES = [
    ("What are the requirements for H-1B specialty occupation?", "Specialty Occupation"),
    ("How does the H-1B lottery and cap work?", "Cap and Lottery"),
    ("Can I change employers while on H-1B visa?", "Portability"),
    ("What is STEM OPT extension for F-1 students?", "OPT"),
    ("How does the employment-based green card process work?", "Green Card"),
    ("What is the prevailing wage for H-1B?", "LCA"),
    ("How long can I stay on H-1B?", "Duration"),
    ("What is H-4 EAD eligibility?", "H-4"),
    ("What is the priority date for green card?", "Priority Date"),
    ("What is AC21 portability?", "Portability"),
]


def detect_dimension(model_name: str) -> int:
    """
    Auto-detect the embedding dimension by loading the model and encoding
    a single sentence. This avoids hardcoding dimensions for each model.

    Common dimensions:
    - 384:  all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5
    - 768:  all-mpnet-base-v2, BAAI/bge-base-en-v1.5, intfloat/e5-base-v2
    - 1024: BAAI/bge-large-en-v1.5, intfloat/e5-large-v2
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Auto-detecting dimension for {model_name}...")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Detected dimension: {dim}")
    return dim


def benchmark_model(model_name: str, dim: int) -> dict:
    sections = _get_fallback_h1b_content()

    store = VectorStore(
        index_dir=f"/tmp/embed_compare_{model_name.replace('/', '_')}",
        embedding_model_name=model_name,
        embedding_dimension=dim,
    )
    ingester = KnowledgeIngester(store)

    # Ingest
    t0 = time.time()
    ingester.ingest_sections(sections, SourceType.USCIS_POLICY_MANUAL)
    ingest_time = time.time() - t0

    # Search
    results_data = []
    total_search_time = 0

    for query, expected_topic in TEST_QUERIES:
        t0 = time.time()
        results = store.search(query, top_k=3)
        search_time = time.time() - t0
        total_search_time += search_time

        top_match = results[0].chunk.context_header if results else "N/A"
        top_score = results[0].score if results else 0
        gap = (results[0].score - results[1].score) if len(results) > 1 else 0
        hit = expected_topic.lower() in top_match.lower()

        results_data.append({
            "query": query,
            "expected": expected_topic,
            "top_match": top_match,
            "top_score": round(top_score, 4),
            "gap": round(gap, 4),
            "hit": hit,
            "scores": [round(r.score, 4) for r in results[:3]],
        })

    hits = sum(1 for r in results_data if r["hit"])
    avg_score = sum(r["top_score"] for r in results_data) / len(results_data)
    avg_gap = sum(r["gap"] for r in results_data) / len(results_data)

    return {
        "model": model_name,
        "dimension": dim,
        "ingest_time_s": round(ingest_time, 2),
        "avg_search_ms": round(total_search_time / len(TEST_QUERIES) * 1000, 1),
        "top1_accuracy": f"{hits}/{len(TEST_QUERIES)}",
        "avg_top_score": round(avg_score, 4),
        "avg_gap": round(avg_gap, 4),
        "details": results_data,
    }


def print_summary_table(all_results: list[dict]):
    """Print a comparison table when multiple models are benchmarked."""
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    header = f"{'Model':<35} {'Dim':>4} {'Top-1':>6} {'Avg Score':>10} {'Avg Gap':>8} {'Ingest':>8} {'Search':>8}"
    print(header)
    print("-" * 90)
    for r in all_results:
        row = (
            f"{r['model']:<35} "
            f"{r['dimension']:>4} "
            f"{r['top1_accuracy']:>6} "
            f"{r['avg_top_score']:>10.4f} "
            f"{r['avg_gap']:>8.4f} "
            f"{r['ingest_time_s']:>7.2f}s "
            f"{r['avg_search_ms']:>6.1f}ms"
        )
        print(row)
    print("=" * 90)


def print_detail(result: dict):
    """Print per-query breakdown for a single model."""
    print(f"\n{'─' * 70}")
    print(f"MODEL: {result['model']} (dim={result['dimension']})")
    print(f"{'─' * 70}")
    for d in result["details"]:
        status = "✓" if d["hit"] else "✗"
        print(f"  {status} Q: {d['query']}")
        print(f"    Top: {d['top_score']:.4f} (gap {d['gap']:.4f}) | {d['top_match']}")
        print(f"    Scores: {d['scores']}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models for RAG retrieval quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s all-MiniLM-L6-v2                          # Single model
  %(prog)s all-MiniLM-L6-v2 all-mpnet-base-v2        # Compare two
  %(prog)s --verbose BAAI/bge-small-en-v1.5           # With per-query details
  %(prog)s --dim 384 my-custom-model                  # Override dimension
        """,
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["all-MiniLM-L6-v2"],
        help="Model name(s) from sentence-transformers or HuggingFace Hub",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Override embedding dimension (auto-detected if omitted)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-query details for each model",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted table",
    )
    args = parser.parse_args()

    all_results = []

    for model_name in args.models:
        print(f"\n>>> Benchmarking: {model_name}")

        # Auto-detect dimension unless overridden
        dim = args.dim if args.dim else detect_dimension(model_name)

        result = benchmark_model(model_name, dim)
        all_results.append(result)

        if args.verbose:
            print_detail(result)

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
