"""Testes unitários das funções de avaliação e gráficos (contrato C4/F4)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from graphids.evaluation import (
    binary_confusion_metrics,
    detection_rate_per_attack,
    extract_embeddings,
    f1_macro,
    plot_anomaly_score_distribution,
    plot_detection_rates,
    plot_pr_curves_comparison,
    plot_precision_recall_curve,
    plot_tsne_projection,
    pr_auc,
    stratified_sample_indices,
)
from graphids.training.run import run_experiment


def test_pr_auc_perfect_and_worst() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(labels, scores) == pytest.approx(1.0)

    scores_bad = [0.9, 0.8, 0.2, 0.1]
    assert pr_auc(labels, scores_bad) < 0.5
    assert pr_auc([], []) == 0.0
    assert pr_auc([0, 0], [0.1, 0.2]) == 0.0


def test_f1_macro_cases() -> None:
    labels = [0, 0, 1, 1]
    preds = [0, 0, 1, 1]
    assert f1_macro(labels, preds) == pytest.approx(1.0)

    preds_half = [0, 1, 0, 1]
    assert f1_macro(labels, preds_half) == pytest.approx(0.5)
    assert f1_macro([], []) == 0.0


def test_detection_rate_per_attack() -> None:
    attack_names = ["Benign", "ddos", "ransomware"]
    attacks = [0, 0, 1, 1, 2, 2]
    scores = [0.1, 0.2, 0.8, 0.4, 0.9, 0.95]
    threshold = 0.5

    rates = detection_rate_per_attack(scores, attacks, threshold, attack_names)

    assert "Benign" not in rates
    assert rates["ddos"] == pytest.approx(0.5)
    assert rates["ransomware"] == pytest.approx(1.0)


def test_detection_rate_empty_class() -> None:
    attack_names = ["Benign", "ddos", "scanning"]
    attacks = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    rates = detection_rate_per_attack(scores, attacks, 0.5, attack_names)

    assert rates["scanning"] == 0.0
    assert rates["ddos"] == pytest.approx(1.0)


def test_binary_confusion_metrics() -> None:
    labels = [0, 0, 0, 1, 1]
    preds = [0, 0, 1, 1, 0]

    res = binary_confusion_metrics(labels, preds)

    assert res["tn"] == 2.0
    assert res["fp"] == 1.0
    assert res["tp"] == 1.0
    assert res["fn"] == 1.0
    assert res["tpr"] == pytest.approx(0.5)
    assert res["fpr"] == pytest.approx(1.0 / 3.0)


def test_stratified_sample_indices_preserves_rare_classes() -> None:
    # Classe 0: 950 itens, Classe 1: 45 itens, Classe 2: 5 itens
    attacks = np.array([0] * 950 + [1] * 45 + [2] * 5)
    idx = stratified_sample_indices(attacks, max_samples=150, seed=42)
    assert len(idx) == 150
    # Todos os 5 elementos da classe rara 2 devem ser preservados
    assert np.sum(attacks[idx] == 2) == 5
    # Classe 1 deve receber cota completa (50 or all 45)
    assert np.sum(attacks[idx] == 1) == 45


def test_plots_and_embeddings_end_to_end(fast_cfg, tmp_path: Path) -> None:
    fast_cfg.num_epochs = 2
    run_experiment(fast_cfg, log=lambda _: None)

    run_dir = Path(fast_cfg.output_dir) / "synthetic" / f"seed{fast_cfg.seed}"
    scores_file = run_dir / "scores_test.npz"
    metrics_file = run_dir / "metrics.json"
    ckpt_file = run_dir / "model.ckpt"

    pr_out = tmp_path / "pr_curve.png"
    fig_pr = plot_precision_recall_curve(scores_file, output_path=pr_out)
    assert pr_out.exists()
    plt.close(fig_pr)

    cmp_out = tmp_path / "pr_cmp.png"
    fig_cmp = plot_pr_curves_comparison(
        {"Synth-A": scores_file, "Synth-B": scores_file}, output_path=cmp_out
    )
    assert cmp_out.exists()
    plt.close(fig_cmp)

    dist_out = tmp_path / "dist.png"
    fig_dist = plot_anomaly_score_distribution(
        scores_file, metrics_path=metrics_file, output_path=dist_out
    )
    assert dist_out.exists()
    plt.close(fig_dist)

    dr_out = tmp_path / "dr.png"
    fig_dr = plot_detection_rates(
        scores_file, metrics_path=metrics_file, output_path=dr_out
    )
    assert dr_out.exists()
    plt.close(fig_dr)

    emb, atk, names = extract_embeddings(
        run_dir / "config.yaml",
        ckpt_file,
        split="test",
        max_samples=60,
        seed=fast_cfg.seed,
        device="cpu",
    )
    assert emb.shape[0] <= 60
    assert emb.shape[1] == fast_cfg.edim_out
    assert len(atk) == emb.shape[0]
    assert len(names) >= 2

    tsne_out = tmp_path / "tsne.png"
    fig_tsne = plot_tsne_projection(
        emb, atk, names, output_path=tsne_out, max_samples=40
    )
    assert tsne_out.exists()
    plt.close(fig_tsne)
