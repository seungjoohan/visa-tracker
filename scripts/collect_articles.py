#!/usr/bin/env python3
"""
Article collection script for building the evaluation dataset.

=== PURPOSE ===

To evaluate our classifiers, we need a "labeled dataset" — a set of articles
where a human has assigned the correct importance level. This script collects
the raw articles; you'll label them later with scripts/label_articles.py.

=== HOW IT WORKS ===

1. Calls the existing NewsAPICollector with two keyword sets:
   - FOCUSED keywords: visa-specific terms that the daily pipeline uses
   - BROAD keywords: immigration-adjacent terms that surface edge cases
     (e.g., "Visa card" about credit cards, "border" about non-visa issues)

2. Runs the keyword-only classifier on each article to generate a "pre_label".
   This pre-label is NOT ground truth — it's a starting point to speed up
   manual labeling. You'll correct it during the labeling step.

3. Appends articles to eval/raw_articles.jsonl, deduplicating by URL.

=== USAGE ===

Run this daily for 7-10 days to accumulate 300+ articles:

    python scripts/collect_articles.py

Check how many you have:

    wc -l eval/raw_articles.jsonl
"""

import sys
import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.news_collector import NewsAPICollector
from app.services.news_analyzer import NewsAnalyzer
from app.core.config import settings
from eval.models import LabeledArticle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FILE = PROJECT_ROOT / "eval" / "raw_articles.jsonl"

# Two keyword sets for diverse article collection
FOCUSED_KEYWORDS = [
    "visa", "immigration", "h1b", "h-1b", "green card",
    "f1 visa", "work permit", "uscis",
]

BROAD_KEYWORDS = [
    "Visa card", "travel ban", "border security",
    "asylum", "deportation", "citizenship",
    "passport", "immigration reform",
]


def load_existing_urls() -> set[str]:
    """Load URLs already in the dataset to avoid duplicates."""
    urls = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        urls.add(record["url"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return urls


async def collect_and_save():
    """Collect articles from NewsAPI and save to JSONL."""

    if not settings.newsapi_key:
        logger.error("NEWSAPI_KEY not set in .env")
        return

    collector = NewsAPICollector(
        api_key=settings.newsapi_key,
        max_articles=100,  # Collect more than the daily pipeline's 50
    )

    # Keyword-only analyzer for pre-labeling (no LLM cost, no semantic model load)
    analyzer = NewsAnalyzer(
        use_semantic=False,
        llm_provider=None,
        retriever=None,
        usage_logger=None,
    )

    existing_urls = load_existing_urls()
    logger.info(f"Existing articles in dataset: {len(existing_urls)}")

    all_articles = []

    # Collect with focused keywords
    logger.info("Collecting with FOCUSED keywords...")
    try:
        focused = await collector.collect_news(FOCUSED_KEYWORDS)
        all_articles.extend(focused)
        logger.info(f"  → {len(focused)} articles")
    except Exception as e:
        logger.error(f"Failed to collect focused articles: {e}")

    # Collect with broad keywords (for edge cases)
    logger.info("Collecting with BROAD keywords...")
    try:
        broad = await collector.collect_news(BROAD_KEYWORDS)
        all_articles.extend(broad)
        logger.info(f"  → {len(broad)} articles")
    except Exception as e:
        logger.error(f"Failed to collect broad articles: {e}")

    if not all_articles:
        logger.warning("No articles collected")
        return

    # Deduplicate and filter out already-saved articles
    new_count = 0
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "a") as f:
        seen_urls = set(existing_urls)  # Copy to track within this batch too

        for article in all_articles:
            url = str(article.url)
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Get pre-label from keyword classifier
            try:
                pre_label = await analyzer.classify_importance(article)
                pre_label_str = pre_label.value
            except Exception:
                pre_label_str = "no_attention_required"

            # Convert to LabeledArticle for consistent serialization
            labeled = LabeledArticle(
                url=url,
                title=article.title,
                description=article.description,
                content=article.content,
                published_at=article.published_at.isoformat(),
                source=article.source.value,
                keywords=article.keywords,
                pre_label=pre_label_str,
                human_label=None,
                labeled_at=None,
            )

            f.write(labeled.model_dump_json() + "\n")
            new_count += 1

    total = len(seen_urls)
    logger.info(f"Added {new_count} new articles (total: {total})")
    print(f"\n{'='*50}")
    print(f"  New articles added:  {new_count}")
    print(f"  Total in dataset:    {total}")
    print(f"  File: {OUTPUT_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(collect_and_save())
