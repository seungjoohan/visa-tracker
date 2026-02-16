"""
Classification metrics calculator.

=== EVALUATION METRICS EXPLAINED ===

When you have a classifier that assigns labels (needs_attention, good_to_know,
no_attention_required), you need to measure how well it performs. Here's what
each metric tells you:

PRECISION: "When the classifier says 'needs_attention', how often is it right?"
  precision = true_positives / (true_positives + false_positives)
  High precision = few false alarms.

RECALL: "Of all articles that truly need attention, how many does the classifier catch?"
  recall = true_positives / (true_positives + false_negatives)
  High recall = few missed items.

F1 SCORE: The harmonic mean of precision and recall.
  f1 = 2 * (precision * recall) / (precision + recall)
  F1 balances both — a classifier that always says "needs_attention" has high
  recall but terrible precision, so its F1 is still low.

MACRO F1: Average F1 across all classes, giving equal weight to each class.
  This is fair when classes are imbalanced (e.g., 10% needs_attention,
  30% good_to_know, 60% no_attention). Accuracy would be misleading
  because a "predict no_attention always" classifier gets 60% accuracy.

CONFUSION MATRIX: A 3x3 grid where cell (i,j) = number of articles whose
  true label is class i but predicted as class j. The diagonal = correct
  predictions. Off-diagonal = errors. Helps you SEE where errors happen.

FALSE NEGATIVE RATE for needs_attention: The most important metric for this
  project. If you miss a "needs_attention" article, the user could miss a
  visa deadline. FNR = false_negatives / (false_negatives + true_positives).
"""

from typing import List, Dict
from eval.models import EvalReport, PerClassMetrics, ClassificationResult

# The three importance classes in display order
CLASS_NAMES = ["needs_attention", "good_to_know", "no_attention_required"]


def compute_metrics(
    classifier_name: str,
    true_labels: List[str],
    predicted_labels: List[str],
    results: List[ClassificationResult],
) -> EvalReport:
    """
    Compute full evaluation metrics from predictions vs. ground truth.

    Args:
        classifier_name: Name of the classifier (for the report)
        true_labels: Human-assigned ground truth labels
        predicted_labels: Classifier's predictions
        results: Per-article ClassificationResult objects (for cost/latency)

    Returns:
        EvalReport with all metrics
    """
    assert len(true_labels) == len(predicted_labels) == len(results), (
        f"Length mismatch: true={len(true_labels)}, "
        f"pred={len(predicted_labels)}, results={len(results)}"
    )

    n = len(true_labels)

    # Build confusion matrix
    # confusion[i][j] = count where true class is CLASS_NAMES[i]
    #                    and predicted class is CLASS_NAMES[j]
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    num_classes = len(CLASS_NAMES)
    confusion = [[0] * num_classes for _ in range(num_classes)]

    for true, pred in zip(true_labels, predicted_labels):
        i = class_to_idx.get(true, -1)
        j = class_to_idx.get(pred, -1)
        if i >= 0 and j >= 0:
            confusion[i][j] += 1

    # Compute per-class metrics from confusion matrix
    per_class: Dict[str, PerClassMetrics] = {}

    for k, class_name in enumerate(CLASS_NAMES):
        # True positives: diagonal element
        tp = confusion[k][k]

        # False positives: column k, excluding diagonal
        # (predicted as class k, but true label is different)
        fp = sum(confusion[i][k] for i in range(num_classes)) - tp

        # False negatives: row k, excluding diagonal
        # (true label is class k, but predicted as something else)
        fn = sum(confusion[k][j] for j in range(num_classes)) - tp

        # Support: total true instances of this class (row sum)
        support = sum(confusion[k][j] for j in range(num_classes))

        # Precision, recall, F1 with zero-division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[class_name] = PerClassMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            support=support,
        )

    # Macro F1: simple average of per-class F1 scores
    # "Macro" = each class counts equally, regardless of how many articles it has
    macro_f1 = sum(m.f1 for m in per_class.values()) / num_classes

    # False negative rate for needs_attention
    # FNR = FN / (FN + TP) = 1 - recall
    na_metrics = per_class.get("needs_attention")
    needs_attention_fnr = 1.0 - na_metrics.recall if na_metrics else 1.0

    # Cost and latency aggregates
    total_latency = sum(r.latency_ms for r in results)
    total_cost = sum(r.cost_usd for r in results)

    return EvalReport(
        classifier_name=classifier_name,
        dataset_size=n,
        per_class=per_class,
        macro_f1=round(macro_f1, 4),
        needs_attention_fnr=round(needs_attention_fnr, 4),
        confusion_matrix=confusion,
        class_names=CLASS_NAMES,
        total_latency_ms=round(total_latency, 2),
        total_cost_usd=round(total_cost, 6),
        avg_latency_ms=round(total_latency / n, 2) if n > 0 else 0.0,
        avg_cost_usd=round(total_cost / n, 6) if n > 0 else 0.0,
    )


def format_report(report: EvalReport) -> str:
    """Format an EvalReport as a readable console string."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Classifier: {report.classifier_name}")
    lines.append(f"  Dataset size: {report.dataset_size}")
    lines.append(f"{'='*60}")

    # Per-class table
    lines.append(f"\n  {'Class':<25} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>9}")
    lines.append(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for name in CLASS_NAMES:
        m = report.per_class[name]
        lines.append(
            f"  {name:<25} {m.precision:>9.4f} {m.recall:>9.4f} "
            f"{m.f1:>9.4f} {m.support:>9d}"
        )
    lines.append(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    lines.append(f"  {'Macro average':<25} {'':>9} {'':>9} {report.macro_f1:>9.4f}")

    # Key metrics
    lines.append(f"\n  needs_attention FNR: {report.needs_attention_fnr:.4f}")
    lines.append(f"  (FNR = false negative rate; lower is better)")

    # Confusion matrix
    lines.append(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = "  " + " " * 25
    for name in CLASS_NAMES:
        short = name[:8]
        header += f" {short:>8}"
    lines.append(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:<25}"
        for j in range(len(CLASS_NAMES)):
            row += f" {report.confusion_matrix[i][j]:>8d}"
        lines.append(row)

    # Cost and latency
    lines.append(f"\n  Total latency: {report.total_latency_ms:.0f}ms")
    lines.append(f"  Avg latency:   {report.avg_latency_ms:.1f}ms/article")
    lines.append(f"  Total cost:    ${report.total_cost_usd:.6f}")
    lines.append(f"  Avg cost:      ${report.avg_cost_usd:.6f}/article")
    lines.append(f"{'='*60}")

    return "\n".join(lines)
