"""Módulo de avaliação e geração de figuras do GraphIDS (Frente 4)."""

from graphids.evaluation.metrics import (
    binary_confusion_metrics,
    detection_rate_per_attack,
    f1_macro,
    pr_auc,
)
from graphids.evaluation.plots import (
    extract_embeddings,
    load_scores,
    plot_anomaly_score_distribution,
    plot_detection_rates,
    plot_pr_curves_comparison,
    plot_precision_recall_curve,
    plot_tsne_projection,
    stratified_sample_indices,
)

__all__ = [
    "binary_confusion_metrics",
    "detection_rate_per_attack",
    "extract_embeddings",
    "f1_macro",
    "load_scores",
    "plot_anomaly_score_distribution",
    "plot_detection_rates",
    "plot_pr_curves_comparison",
    "plot_precision_recall_curve",
    "plot_tsne_projection",
    "pr_auc",
    "stratified_sample_indices",
]
