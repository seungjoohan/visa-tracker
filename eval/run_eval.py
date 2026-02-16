#!/usr/bin/env python3
"""
Evaluation CLI — run classifiers against the labeled dataset.

=== WHAT THIS DOES ===

Takes a labeled dataset (articles with human-assigned ground truth labels)
and runs one or more classifiers against it. For each classifier, it:

1. Classifies every article
2. Compares predictions against ground truth
3. Computes precision, recall, F1, confusion matrix
4. Prints a report and saves results to eval/results/

=== USAGE ===

# Run free classifiers (no API cost)
python eval/run_eval.py --classifier keyword,semantic

# Run LLM classifiers (costs ~$0.0002/article, requires --confirm-cost)
python eval/run_eval.py --classifier llm,llm_rag --confirm-cost

# Quick test with just 10 articles
python eval/run_eval.py --classifier keyword --limit 10

# Custom dataset path
python eval/run_eval.py --classifier keyword --dataset eval/labeled_articles.jsonl

=== COST SAFETY ===

LLM classifiers (llm, llm_rag) require the --confirm-cost flag. Before
running, the script estimates the total cost and asks you to confirm.
At ~$0.0002/article with Haiku, 300 articles costs ~$0.06 — cheap,
but the safeguard teaches responsible LLM usage.
"""

import sys
import os
import json
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.models import LabeledArticle, EvalReport
from eval.classifiers import CLASSIFIERS, EvalClassifier
from eval.metrics import compute_metrics, format_report

logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO logs from models loading
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "labeled_articles.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def load_labeled_dataset(filepath: Path) -> list[LabeledArticle]:
    """Load labeled articles from JSONL. Only returns articles with human labels."""
    articles = []
    if not filepath.exists():
        return articles

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                article = LabeledArticle.model_validate_json(line)
                if article.human_label:  # Only include labeled articles
                    articles.append(article)
            except Exception:
                continue

    return articles


async def run_classifier(
    classifier: EvalClassifier,
    articles: list[LabeledArticle],
) -> EvalReport:
    """Run one classifier on all articles and return the evaluation report."""

    print(f"\n  Running {classifier.name} classifier on {len(articles)} articles...")
    await classifier.setup()

    true_labels = []
    predicted_labels = []
    results = []

    for i, article in enumerate(articles, 1):
        if i % 50 == 0 or i == len(articles):
            print(f"    Progress: {i}/{len(articles)}", end="\r")

        result = await classifier.classify(article)
        true_labels.append(article.human_label)
        predicted_labels.append(result.predicted_label)
        results.append(result)

    print()  # Clear the progress line

    await classifier.teardown()

    report = compute_metrics(
        classifier_name=classifier.name,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        results=results,
    )

    return report


def save_report(report: EvalReport):
    """Save evaluation report to eval/results/ as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{report.classifier_name}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    with open(filepath, "w") as f:
        f.write(report.model_dump_json(indent=2))

    print(f"  Saved: {filepath}")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classifiers against the labeled dataset"
    )
    parser.add_argument(
        "--classifier",
        required=True,
        help="Comma-separated classifier names: keyword,semantic,llm,llm_rag",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to labeled JSONL dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of articles (0 = all, useful for testing)",
    )
    parser.add_argument(
        "--confirm-cost",
        action="store_true",
        help="Required for LLM classifiers (acknowledges API cost)",
    )
    args = parser.parse_args()

    # Parse classifier names
    classifier_names = [c.strip() for c in args.classifier.split(",")]
    for name in classifier_names:
        if name not in CLASSIFIERS:
            print(f"Unknown classifier: {name}")
            print(f"Available: {', '.join(CLASSIFIERS.keys())}")
            sys.exit(1)

    # Check if LLM classifiers need cost confirmation
    llm_classifiers = [n for n in classifier_names if CLASSIFIERS[n].requires_llm]
    if llm_classifiers and not args.confirm_cost:
        print(f"\n  LLM classifiers requested: {', '.join(llm_classifiers)}")
        print(f"  Estimated cost: ~$0.0002/article × dataset size")
        print(f"  Add --confirm-cost to proceed.")
        sys.exit(1)

    # Load dataset
    articles = load_labeled_dataset(args.dataset)
    if not articles:
        print(f"No labeled articles found in {args.dataset}")
        print("Run 'python scripts/collect_articles.py' then 'python scripts/label_articles.py'.")
        sys.exit(1)

    if args.limit > 0:
        articles = articles[:args.limit]

    print(f"\n{'='*60}")
    print(f"  Evaluation Run")
    print(f"{'='*60}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Articles: {len(articles)}")
    print(f"  Classifiers: {', '.join(classifier_names)}")

    # Run each classifier
    for name in classifier_names:
        classifier_cls = CLASSIFIERS[name]
        classifier = classifier_cls()

        try:
            report = await run_classifier(classifier, articles)
            print(format_report(report))
            save_report(report)
        except Exception as e:
            print(f"\n  ERROR running {name}: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
