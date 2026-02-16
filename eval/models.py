"""
Data models for the evaluation framework.

=== THREE MODELS, THREE PURPOSES ===

1. LabeledArticle: A news article with a human-assigned ground truth label.
   This is what you get after collecting articles and manually labeling them.
   It's intentionally flat (no nested Pydantic models) so it serializes
   cleanly to JSONL.

2. ClassificationResult: The output of running one classifier on one article.
   Captures the prediction, timing, cost, and confidence — everything needed
   to compute metrics AND compare cost efficiency.

3. EvalReport: The aggregated result of running one classifier on many articles.
   Contains per-class precision/recall/F1, confusion matrix, and cost totals.
   This is what gets saved to eval/results/ and consumed by compare.py.
"""

from pydantic import BaseModel
from typing import Dict, List, Optional


class LabeledArticle(BaseModel):
    """
    A news article with ground truth label for evaluation.

    Fields mirror NewsArticle but simplified for serialization.
    The key field is `human_label` — the manually-assigned ground truth
    that classifiers will be measured against.
    """
    url: str
    title: str
    description: Optional[str] = None
    content: Optional[str] = None
    published_at: str            # ISO format string (not datetime — simpler serialization)
    source: str                  # "news_api" or "white_house" (string, not enum)
    keywords: List[str] = []
    pre_label: Optional[str] = None    # Auto-assigned by keyword classifier during collection
    human_label: Optional[str] = None  # Ground truth: "needs_attention" | "good_to_know" | "no_attention_required"
    labeled_at: Optional[str] = None   # ISO timestamp of when human labeled it


class ClassificationResult(BaseModel):
    """
    Result of classifying one article with one classifier.

    === WHY TRACK ALL THIS PER-ARTICLE? ===
    - predicted_label: For computing precision/recall/F1
    - latency_ms: For speed comparison (keyword < semantic < LLM)
    - cost_usd: For cost comparison ($0 for keyword/semantic, ~$0.0002 for LLM)
    - confidence: LLM classifiers report confidence (0-1), others don't
    """
    predicted_label: str         # "needs_attention" | "good_to_know" | "no_attention_required"
    latency_ms: float
    cost_usd: float = 0.0       # Non-LLM classifiers cost $0
    confidence: Optional[float] = None  # From AnalysisResult.confidence (LLM only)


class PerClassMetrics(BaseModel):
    """Precision/recall/F1 for one class."""
    precision: float
    recall: float
    f1: float
    support: int     # Number of true instances of this class


class EvalReport(BaseModel):
    """
    Full evaluation report for one classifier on one dataset.

    === WHAT'S IN HERE? ===
    - per_class: Precision/recall/F1 for each of the 3 importance levels
    - macro_f1: Unweighted average of per-class F1 scores. Treats all classes
      equally regardless of how many articles are in each class.
    - needs_attention_fnr: False negative rate for the most critical class.
      If this is 0.20, we're missing 20% of truly urgent articles — bad!
    - confusion_matrix: 3x3 grid. Row i, column j = count of articles
      whose true label is class i but predicted as class j.
    - Cost/latency totals for comparing efficiency across classifiers.
    """
    classifier_name: str
    dataset_size: int
    per_class: Dict[str, PerClassMetrics]
    macro_f1: float
    needs_attention_fnr: float   # False negative rate for needs_attention
    confusion_matrix: List[List[int]]  # 3x3
    class_names: List[str]       # Labels for confusion matrix axes
    total_latency_ms: float
    total_cost_usd: float
    avg_latency_ms: float
    avg_cost_usd: float
