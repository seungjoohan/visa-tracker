#!/usr/bin/env python3
"""
Terminal-based article labeling tool.

=== PURPOSE ===

After collecting articles with collect_articles.py, you need to assign
ground truth labels. This tool shows each unlabeled article and lets you
pick the correct importance level.

=== HOW LABELING WORKS ===

For each article, you see:
  - Title
  - Description (first 200 chars)
  - Keywords found
  - Pre-label (what the keyword classifier guessed)

Then you choose:
  [1] needs_attention       — Urgent: deadlines, bans, new requirements
  [2] good_to_know          — Useful: policy changes, updates, proposals
  [3] no_attention_required  — Background: general info, not actionable
  [s] skip                  — Not sure, come back later
  [q] quit                  — Save and exit

Your label is the "ground truth" that classifiers will be measured against.
Be consistent — if in doubt, think: "Would I want to be emailed about this?"
  - Yes, urgently → needs_attention
  - Yes, eventually → good_to_know
  - No → no_attention_required

=== USAGE ===

    python scripts/label_articles.py

    # Resume from where you left off (already labeled ones are skipped)
    python scripts/label_articles.py

    # Only show articles with a specific pre-label
    python scripts/label_articles.py --filter needs_attention
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.models import LabeledArticle

PROJECT_ROOT = Path(__file__).parent.parent
RAW_FILE = PROJECT_ROOT / "eval" / "raw_articles.jsonl"
LABELED_FILE = PROJECT_ROOT / "eval" / "labeled_articles.jsonl"

VALID_LABELS = {"1": "needs_attention", "2": "good_to_know", "3": "no_attention_required"}


def load_articles(filepath: Path) -> list[LabeledArticle]:
    """Load articles from a JSONL file."""
    articles = []
    if filepath.exists():
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        articles.append(LabeledArticle.model_validate_json(line))
                    except Exception as e:
                        print(f"  [warn] Skipping malformed line: {e}")
    return articles


def load_labeled_urls() -> set[str]:
    """Get URLs that already have human labels."""
    urls = set()
    if LABELED_FILE.exists():
        with open(LABELED_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if record.get("human_label"):
                            urls.add(record["url"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return urls


def save_labeled(article: LabeledArticle):
    """Append one labeled article to the labeled file."""
    LABELED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELED_FILE, "a") as f:
        f.write(article.model_dump_json() + "\n")


def display_article(article: LabeledArticle, index: int, total: int):
    """Display an article for labeling."""
    print(f"\n{'─'*60}")
    print(f"  Article {index}/{total}")
    print(f"{'─'*60}")
    print(f"  Title:    {article.title}")

    if article.description:
        desc = article.description[:200]
        if len(article.description) > 200:
            desc += "..."
        print(f"  Desc:     {desc}")

    if article.keywords:
        print(f"  Keywords: {', '.join(article.keywords[:5])}")

    print(f"  Date:     {article.published_at[:10]}")
    print(f"  Source:   {article.source}")

    pre_label_display = article.pre_label or "none"
    print(f"  Pre-label: {pre_label_display}")
    print()


def get_label_input() -> str | None:
    """Prompt user for a label. Returns label string, 'skip', or None (quit)."""
    print("  [1] needs_attention")
    print("  [2] good_to_know")
    print("  [3] no_attention_required")
    print("  [s] skip    [q] quit")
    print()

    while True:
        choice = input("  Your label: ").strip().lower()
        if choice == "q":
            return None
        if choice == "s":
            return "skip"
        if choice in VALID_LABELS:
            return VALID_LABELS[choice]
        print("  Invalid input. Enter 1, 2, 3, s, or q.")


def main():
    parser = argparse.ArgumentParser(description="Label articles for evaluation")
    parser.add_argument(
        "--filter",
        choices=["needs_attention", "good_to_know", "no_attention_required"],
        help="Only show articles with this pre-label",
    )
    args = parser.parse_args()

    # Load raw articles and find unlabeled ones
    raw_articles = load_articles(RAW_FILE)
    if not raw_articles:
        print(f"No articles found in {RAW_FILE}")
        print("Run 'python scripts/collect_articles.py' first.")
        return

    labeled_urls = load_labeled_urls()

    # Filter to unlabeled articles
    unlabeled = [a for a in raw_articles if a.url not in labeled_urls]

    # Apply optional filter
    if args.filter:
        unlabeled = [a for a in unlabeled if a.pre_label == args.filter]

    total_raw = len(raw_articles)
    total_labeled = len(labeled_urls)
    total_unlabeled = len(unlabeled)

    print(f"\n{'='*60}")
    print(f"  Article Labeling Tool")
    print(f"{'='*60}")
    print(f"  Total articles:    {total_raw}")
    print(f"  Already labeled:   {total_labeled}")
    print(f"  To label:          {total_unlabeled}")
    if args.filter:
        print(f"  Filter:            {args.filter}")
    print(f"{'='*60}")

    if total_unlabeled == 0:
        print("\n  All articles are labeled! Nothing to do.")
        return

    # Labeling loop
    session_count = 0

    for i, article in enumerate(unlabeled, 1):
        display_article(article, i, total_unlabeled)
        label = get_label_input()

        if label is None:
            # User quit
            break

        if label == "skip":
            continue

        # Save the labeled article
        article.human_label = label
        article.labeled_at = datetime.utcnow().isoformat()
        save_labeled(article)
        session_count += 1
        print(f"  ✓ Labeled as: {label}  (session total: {session_count})")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Session complete")
    print(f"  Labeled this session: {session_count}")
    print(f"  Total labeled:        {total_labeled + session_count}")
    print(f"  Saved to: {LABELED_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
