"""Model evaluation and fairness analysis."""

from acr.evaluation.metrics import compute_classification_metrics
from acr.evaluation.fairness import compute_fairness_metrics

__all__ = ["compute_classification_metrics", "compute_fairness_metrics"]
