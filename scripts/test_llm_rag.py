#!/usr/bin/env python3
"""
Test script for the full LLM + RAG pipeline.

Tests two paths:
1. Full pipeline: Retriever + LLM (Claude) -> AnalysisResult
2. Fallback path: No LLM -> keyword+semantic classification
"""

import asyncio
import sys
import os
import logging

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from app.models.news import NewsArticle, NewsSource, ImportanceLevel
from app.models.llm import AnalysisResult
from app.services.news_analyzer import NewsAnalyzer
from app.core.config import settings
from app.core.dependencies import get_llm_provider, get_retriever, get_llm_usage_logger

# Configure logging so we can see the pipeline steps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_fake_article() -> NewsArticle:
    """Create a realistic fake H-1B news article for testing."""
    return NewsArticle(
        title="USCIS Announces New H-1B Registration Process for FY2026",
        description=(
            "The U.S. Citizenship and Immigration Services announced sweeping changes "
            "to the H-1B electronic registration system for the FY2026 cap season, "
            "including a new beneficiary-centric selection process designed to reduce "
            "duplicate registrations and improve fairness."
        ),
        content=(
            "WASHINGTON -- USCIS today published a final rule overhauling the H-1B "
            "electronic registration system effective March 2026. Key changes include: "
            "(1) A beneficiary-centric selection process that limits each unique "
            "beneficiary to a single entry in the lottery, regardless of how many "
            "employers submit registrations on their behalf. "
            "(2) A new requirement for registrants to provide valid passport or "
            "travel document information at the registration stage. "
            "(3) Increased registration fees from $10 to $215 per beneficiary. "
            "(4) An organizational accounts system requiring companies with 25 or "
            "more registrations to use a dedicated USCIS online account. "
            "The initial registration period will open on March 7, 2026, and run "
            "through March 24, 2026. Employers and their attorneys should begin "
            "preparing beneficiary documentation now. USCIS expects these changes "
            "to significantly reduce the number of duplicate registrations that "
            "have plagued the system in recent years, where some beneficiaries "
            "had dozens of registrations filed on their behalf."
        ),
        url="https://www.uscis.gov/news/alerts/uscis-announces-new-h1b-registration-fy2026",
        published_at=datetime(2026, 1, 15, 14, 0, 0),
        source=NewsSource.NEWS_API,
        keywords=["h-1b", "uscis", "registration", "visa", "lottery"],
        importance_level=None,
    )


async def test_full_pipeline():
    """Test 1: Full RAG pipeline with LLM."""
    print("\n" + "=" * 70)
    print("TEST 1: Full LLM + RAG Pipeline")
    print("=" * 70)

    llm_provider = get_llm_provider()
    retriever = get_retriever()
    usage_logger = get_llm_usage_logger()

    print(f"  LLM provider : {llm_provider.__class__.__name__ if llm_provider else 'None (disabled)'}")
    print(f"  LLM model    : {settings.llm_model}")
    print(f"  Retriever    : {'loaded' if retriever else 'None (no index)'}")
    print(f"  Usage logger : {usage_logger.__class__.__name__}")

    if not llm_provider:
        print("\n  [SKIP] LLM is disabled (use_llm=False or no ANTHROPIC_API_KEY).")
        print("         Set USE_LLM=true and ANTHROPIC_API_KEY in .env to enable.")
        return

    analyzer = NewsAnalyzer(
        llm_provider=llm_provider,
        retriever=retriever,
        usage_logger=usage_logger,
    )

    article = create_fake_article()
    print(f"\n  Article: {article.title}")
    print(f"  Source : {article.source.value}")
    print(f"  Date   : {article.published_at.isoformat()}")

    print("\n  Calling analyze_article() ...")
    result: AnalysisResult | None = await analyzer.analyze_article(article)

    if result is None:
        print("\n  [WARN] analyze_article() returned None (LLM call or parsing failed).")
        print("         Check logs above for details.")
        return

    print("\n  --- AnalysisResult ---")
    print(f"  importance          : {result.importance}")
    print(f"  affected_visa_types : {result.affected_visa_types}")
    print(f"  action_required     : {result.action_required}")
    print(f"  deadline            : {result.deadline}")
    print(f"  impact_summary      : {result.impact_summary}")
    print(f"  relevant_policy_refs: {result.relevant_policy_refs}")
    print(f"  confidence          : {result.confidence}")

    # Also test classify_importance which uses analyze_article under the hood
    print("\n  Calling classify_importance() (should use cached result) ...")
    level = await analyzer.classify_importance(article)
    print(f"  ImportanceLevel     : {level.value}")

    # And generate_summary
    print("\n  Calling generate_summary() (should use cached result) ...")
    summary = await analyzer.generate_summary(article)
    print(f"  Summary             : {summary}")

    print("\n  [OK] Full pipeline test complete.")


async def test_fallback_path():
    """Test 2: Fallback path without LLM."""
    print("\n" + "=" * 70)
    print("TEST 2: Fallback Path (No LLM)")
    print("=" * 70)

    analyzer = NewsAnalyzer(
        use_semantic=False,  # also skip semantic to keep it fast
        llm_provider=None,
        retriever=None,
        usage_logger=None,
    )

    article = create_fake_article()
    print(f"\n  Article: {article.title}")

    print("\n  Calling analyze_article() without LLM ...")
    result = await analyzer.analyze_article(article)
    print(f"  Result: {result}  (expected: None)")

    print("\n  Calling classify_importance() (keyword-only fallback) ...")
    level = await analyzer.classify_importance(article)
    print(f"  ImportanceLevel: {level.value}")

    print("\n  Calling generate_summary() (stub fallback) ...")
    summary = await analyzer.generate_summary(article)
    print(f"  Summary: {summary}")

    print("\n  Calling calculate_relevance_score() ...")
    score = await analyzer.calculate_relevance_score(article)
    print(f"  Relevance score: {score:.4f}")

    print("\n  [OK] Fallback path test complete.")


async def main():
    print("=" * 70)
    print("  visa-tracker: LLM + RAG Pipeline Test")
    print("=" * 70)
    print(f"  Settings:")
    print(f"    use_llm          = {settings.use_llm}")
    print(f"    anthropic_api_key = {'***' + settings.anthropic_api_key[-4:] if settings.anthropic_api_key else '(not set)'}")
    print(f"    llm_model        = {settings.llm_model}")
    print(f"    embedding_model  = {settings.embedding_model}")
    print(f"    retrieval_top_k  = {settings.retrieval_top_k}")
    print(f"    prompts_dir      = {settings.prompts_dir}")

    await test_full_pipeline()
    await test_fallback_path()

    print("\n" + "=" * 70)
    print("  All tests finished.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
