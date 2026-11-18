"""Testes das métricas (casos calculados à mão) e das figuras (dados sintéticos)."""

import json

import numpy as np
import pytest

from graphids.evaluation import (
    attack_counts,
    confusion,
    detection_rate_per_attack,
    f1_macro,
    plots,
    pr_auc,
    predict,
)


def test_pr_auc_perfect_and_hand_case():
    labels = np.array([0, 0, 1, 1])
    assert pr_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    # Ordem por score: 1(0.9), 0(0.8), 1(0.7), 0(0.1)
    # AP = 0.5 * 1/1 + 0.5 * 2/3 = 0.8333...
    assert pr_auc(labels[[1, 2, 0, 3]], np.array([0.8, 0.7, 0.1, 0.9])) == (
        pytest.approx(5 / 6)
    )


def test_f1_macro_hand_case():
    labels = np.array([0, 0, 0, 1])
    preds = np.array([0, 0, 1, 1])
    # classe 0: P=1, R=2/3 -> F1=0.8 ; classe 1: P=0.5, R=1 -> F1=2/3
    assert f1_macro(labels, preds) == pytest.approx((0.8 + 2 / 3) / 2)


def test_predict_is_strict_and_confusion():
    preds = predict(np.array([0.5, 1.0, 1.5]), threshold=1.0)
    assert preds.tolist() == [0, 0, 1]
    cm = confusion(np.array([0, 1, 1]), preds)
    assert cm.tolist() == [[1, 0], [1, 1]]


def test_detection_rate_per_attack():
    scores = np.array([0.1, 2.0, 3.0, 0.5, 5.0, 0.2])
    attacks = np.array([0, 0, 1, 1, 2, 0])
    names = ["Benign", "DoS", "Scan", "Nunca"]
    rates = detection_rate_per_attack(scores, attacks, 1.0, names)
    assert rates == {"Benign": pytest.approx(1 / 3), "DoS": 0.5, "Scan": 1.0}
    assert attack_counts(attacks, names) == {"Benign": 3, "DoS": 2, "Scan": 1}


def _fake_run(seed_dir, seed, rng):
    n = 400
    attacks = rng.integers(0, 3, n)
    labels = (attacks > 0).astype(int)
    errors = rng.lognormal(mean=labels * 2.0, sigma=0.5).astype(np.float32)
    seed_dir.mkdir(parents=True)
    np.savez(
        seed_dir / "scores_test.npz",
        errors=errors,
        labels=labels,
        attacks=attacks,
        attack_names=np.array(["Benign", "DoS", "Scan"]),
    )
    (seed_dir / "metrics.json").write_text(
        json.dumps({"dataset": "fake", "seed": seed, "threshold": 2.5})
    )
    hist = [
        {"epoch": e, "train_loss": 1 / e, "val_loss": 1.2 / e, "val_pr_auc": 0.9}
        for e in range(1, 4)
    ]
    (seed_dir / "history.json").write_text(json.dumps(hist))


def test_all_figures_from_synthetic_npz(tmp_path):
    rng = np.random.default_rng(0)
    for s in (1, 2, 3):
        _fake_run(tmp_path / "runs" / "fake" / f"seed{s}", s, rng)
    # Seed incompleta (sem metrics.json) deve ser ignorada.
    (tmp_path / "runs" / "fake" / "seed9").mkdir()

    runs = plots.load_runs(tmp_path / "runs" / "fake")
    assert [r.seed for r in runs] == [1, 2, 3]

    paths = plots.all_figures(tmp_path / "runs" / "fake", tmp_path / "fig")
    assert len(paths) == 4
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)

    table = plots.detection_table(runs)
    assert set(table) == {"Benign", "DoS", "Scan"}
    assert all(0.0 <= m <= 1.0 for m, _ in table.values())


def test_tsne_helpers_on_synthetic_model(tmp_path):
    import torch

    from graphids.config import load_config
    from graphids.data import load_splits
    from graphids.evaluation.embeddings import (
        Projection,
        flow_embeddings,
        plot_projections,
        stratified_sample,
        transformer_latents,
        tsne_2d,
    )
    from graphids.models.graphids import GraphIDS

    cfg = load_config("configs/synthetic.yaml")
    splits = load_splits(cfg.data_config())
    model = GraphIDS.from_config(cfg, splits.num_features)
    attacks = splits.test.edge_attack.numpy()

    ids = stratified_sample(attacks, per_class=40, seed=0)
    counts = np.bincount(attacks[ids])
    assert counts.max() <= 40 and len(np.unique(attacks[ids])) == len(
        np.unique(attacks)
    )

    emb = flow_embeddings(model, splits.test, ids, cfg.fanout, batch_size=50)
    assert emb.shape == (len(ids), cfg.edim_out)
    # Mesmo resultado em outro tamanho de batch: a vizinhança é a do grafo todo.
    torch.testing.assert_close(
        torch.from_numpy(emb),
        torch.from_numpy(flow_embeddings(model, splits.test, ids, cfg.fanout)),
    )
    lat = transformer_latents(model, emb, cfg.window_size)
    assert lat.shape == (len(ids), cfg.embed_dim)

    xy = tsne_2d(emb)
    assert xy.shape == (len(ids), 2)
    path = plot_projections(
        [
            Projection("a", xy, attacks[ids]),
            Projection("b", tsne_2d(lat), attacks[ids]),
        ],
        list(splits.test.attack_names),
        "teste",
        tmp_path / "t.pdf",
    )
    assert path.stat().st_size > 0
