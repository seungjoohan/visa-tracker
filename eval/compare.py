#!/usr/bin/env python3
"""
Comparison CLI — side-by-side comparison of evaluation results.

=== WHAT THIS DOES ===

Reads saved EvalReport JSON files from eval/results/ and produces a
side-by-side comparison table. This is the "money shot" for your portfolio:
it shows how each classifier method performs across all metrics.

=== USAGE ===

# Compare all saved results
python eval/compare.py

# Specify a different results directory
python eval/compare.py --results-dir eval/results/

=== WHAT YOU'RE LOOKING FOR ===

The comparison should tell a story:
1. Keyword is fast and free but misses nuance (low F1)
2. Semantic catches more but still makes mistakes
3. LLM is better but sometimes hallucinates
4. LLM+RAG is best because grounding in policy docs reduces errors

The "needs_attention" FNR is the most important column — that's how
many urgent articles each method MISSES.
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.models import EvalReport

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def load_reports(results_dir: Path) -> list[EvalReport]:
    """
    Load all EvalReport JSON files from the results directory.

    If multiple reports exist for the same classifier, uses the most recent
    one (based on filename timestamp).
    """
    reports_by_name: dict[str, tuple[str, EvalReport]] = {}

    if not results_dir.exists():
        return []

    for filepath in sorted(results_dir.glob("*.json")):
        try:
            with open(filepath) as f:
                data = json.load(f)
            report = EvalReport.model_validate(data)
            # Keep the most recent report per classifier
            # (files are sorted alphabetically, so later timestamps win)
            reports_by_name[report.classifier_name] = (filepath.name, report)
        except Exception as e:
            print(f"  [warn] Skipping {filepath.name}: {e}")

    # Sort by: keyword → semantic → llm → llm_rag
    order = {"keyword": 0, "semantic": 1, "llm": 2, "llm_rag": 3}
    sorted_reports = sorted(
        reports_by_name.values(),
        key=lambda x: order.get(x[1].classifier_name, 99),
    )

    return [report for _, report in sorted_reports]


def format_comparison_table(reports: list[EvalReport]) -> str:
    """
    Format a side-by-side comparison table.

    The table shows key metrics for each classifier in a compact format
    that's easy to scan and include in documentation.
    """
    if not reports:
        return "  No reports found."

    lines = []

    # Header
    lines.append(f"\n{'='*80}")
    lines.append(f"  Classifier Comparison")
    lines.append(f"{'='*80}")

    # Main comparison table
    header = f"  {'Classifier':<12} {'Macro F1':>9} {'NA F1':>8} {'NA FNR':>8} {'Avg ms':>9} {'$/article':>11} {'Total $':>9}"
    sep =    f"  {'─'*12} {'─'*9} {'─'*8} {'─'*8} {'─'*9} {'─'*11} {'─'*9}"

    lines.append(header)
    lines.append(sep)

    for r in reports:
        na_f1 = r.per_class.get("needs_attention")
        na_f1_str = f"{na_f1.f1:.4f}" if na_f1 else "N/A"

        lines.append(
            f"  {r.classifier_name:<12} "
            f"{r.macro_f1:>9.4f} "
            f"{na_f1_str:>8} "
            f"{r.needs_attention_fnr:>8.4f} "
            f"{r.avg_latency_ms:>9.1f} "
            f"${r.avg_cost_usd:>10.6f} "
            f"${r.total_cost_usd:>8.4f}"
        )

    lines.append(sep)

    # Per-class breakdown for each classifier
    lines.append(f"\n  Per-Class F1 Scores:")
    class_header = f"  {'Classifier':<12}"
    for cls in ["needs_attention", "good_to_know", "no_attention_req"]:
        class_header += f" {cls[:15]:>15}"
    lines.append(class_header)
    lines.append(f"  {'─'*12} {'─'*15} {'─'*15} {'─'*15}")

    for r in reports:
        row = f"  {r.classifier_name:<12}"
        for cls in ["needs_attention", "good_to_know", "no_attention_required"]:
            m = r.per_class.get(cls)
            row += f" {m.f1:>15.4f}" if m else f" {'N/A':>15}"
        lines.append(row)

    # Confusion matrices
    for r in reports:
        lines.append(f"\n  Confusion Matrix: {r.classifier_name}")
        lines.append(f"  (rows=true, cols=predicted)")
        header = "  " + " " * 25
        for name in r.class_names:
            header += f" {name[:8]:>8}"
        lines.append(header)
        for i, name in enumerate(r.class_names):
            row = f"  {name:<25}"
            for j in range(len(r.class_names)):
                row += f" {r.confusion_matrix[i][j]:>8d}"
            lines.append(row)

    lines.append(f"\n{'='*80}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results across classifiers"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory with saved eval reports (default: {DEFAULT_RESULTS_DIR})",
    )
    args = parser.parse_args()

    reports = load_reports(args.results_dir)

    if not reports:
        print(f"\n  No evaluation results found in {args.results_dir}")
        print(f"  Run 'python eval/run_eval.py --classifier keyword' first.")
        return

    print(f"\n  Loaded {len(reports)} report(s) from {args.results_dir}")
    for r in reports:
        print(f"    - {r.classifier_name}: {r.dataset_size} articles")

    print(format_comparison_table(reports))


if __name__ == "__main__":
    main()
