"""Figuras do relatório (entrega 4.3 da F4).

Tudo é lido dos artefatos do contrato C2 em ``results/runs/<dataset>/seed<s>/``
(``scores_test.npz``, ``metrics.json`` e ``history.json``), então as figuras
podem ser refeitas a qualquer momento sem treinar de novo.

Figuras geradas (equivalentes às do artigo):

- ``pr_curves``: curva precisão-revocação no teste, uma linha por seed
  (Figura 4);
- ``score_distribution``: distribuição do score de anomalia por categoria, em
  escala log, com o threshold escolhido na validação (Figuras 5 a 8);
- ``detection_rates``: taxa de detecção por categoria de ataque, média ± desvio
  entre seeds (complementa as Figuras 5 a 8);
- ``training_curves``: perda de treino/validação e PR-AUC de validação por
  época (diagnóstico de convergência).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display: roda no servidor e nos testes

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from graphids.evaluation.metrics import (  # noqa: E402
    detection_rate_per_attack,
    pr_auc,
)

# Paleta neutra e legível em impressão P&B (seeds em tons distintos).
SEED_COLORS = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#bf9000"]
THRESHOLD_COLOR = "#c00000"


@dataclass
class SeedRun:
    """Artefatos de uma execução (um dataset, uma seed)."""

    dataset: str
    seed: int
    errors: np.ndarray
    labels: np.ndarray
    attacks: np.ndarray
    attack_names: list[str]
    threshold: float
    history: list[dict]


def load_runs(dataset_dir: str | Path, split: str = "test") -> list[SeedRun]:
    """Carrega todas as seeds de ``results/runs/<dataset>/`` (ordenadas por seed)."""
    dataset_dir = Path(dataset_dir)
    runs = []
    for seed_dir in sorted(dataset_dir.glob("seed*")):
        scores_file = seed_dir / f"scores_{split}.npz"
        metrics_file = seed_dir / "metrics.json"
        if not (scores_file.exists() and metrics_file.exists()):
            continue  # execução incompleta
        with np.load(scores_file, allow_pickle=False) as d:
            errors, labels = d["errors"], d["labels"]
            attacks, names = d["attacks"], [str(n) for n in d["attack_names"]]
        metrics = json.loads(metrics_file.read_text())
        hist_file = seed_dir / "history.json"
        history = json.loads(hist_file.read_text()) if hist_file.exists() else []
        runs.append(
            SeedRun(
                dataset=metrics.get("dataset", dataset_dir.name),
                seed=int(metrics.get("seed", seed_dir.name.removeprefix("seed"))),
                errors=errors,
                labels=labels,
                attacks=attacks,
                attack_names=names,
                threshold=float(metrics["threshold"]),
                history=history,
            )
        )
    return sorted(runs, key=lambda r: r.seed)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def pr_curves(runs: list[SeedRun], path: str | Path) -> Path:
    """Curva precisão-revocação de cada seed, com a prevalência como referência.

    A linha tracejada é a precisão de um classificador aleatório (a fração de
    ataques no teste): em datasets com muitos ataques ela é alta, o que muda a
    leitura da PR-AUC (ponto discutido na justificativa do NF-ToN-IoT-v3).
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    for i, run in enumerate(runs):
        prec, rec, _ = precision_recall_curve(run.labels, run.errors)
        ap = pr_auc(run.labels, run.errors)
        ax.plot(
            rec,
            prec,
            color=SEED_COLORS[i % len(SEED_COLORS)],
            lw=1.4,
            label=f"seed {run.seed} (PR-AUC {ap:.4f})",
        )
    prevalence = float(np.mean(runs[0].labels))
    ax.axhline(prevalence, color="grey", ls="--", lw=1, label="aleatório")
    ax.set(
        xlabel="Revocação",
        ylabel="Precisão",
        xlim=(0, 1.01),
        ylim=(0, 1.02),
        title=runs[0].dataset,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, Path(path))


def score_distribution(run: SeedRun, path: str | Path) -> Path:
    """Boxplot do score de anomalia por categoria (log), com o threshold.

    Categorias ordenadas pela mediana do score; o rótulo do eixo traz o número
    de fluxos de cada uma no teste.
    """
    present = [i for i in range(len(run.attack_names)) if np.any(run.attacks == i)]
    data = [run.errors[run.attacks == i] for i in present]
    order = np.argsort([np.median(d) for d in data])
    data = [np.clip(data[k], 1e-8, None) for k in order]  # log não aceita 0
    labels = [
        f"{run.attack_names[present[k]]} ({int(np.sum(run.attacks == present[k]))})"
        for k in order
    ]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(data) + 1.2))
    ax.boxplot(
        data,
        orientation="horizontal",
        whis=(1, 99),
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#dbe5f1", "edgecolor": "#1f4e79"},
        medianprops={"color": "#1f4e79"},
    )
    ax.set_yticks(range(1, len(labels) + 1), labels, fontsize=8)
    ax.set_xscale("log")
    ax.axvline(
        run.threshold,
        color=THRESHOLD_COLOR,
        ls="--",
        lw=1.2,
        label=f"threshold = {run.threshold:.3g}",
    )
    ax.set(
        xlabel="Score de anomalia (erro de reconstrução, log)",
        title=f"{run.dataset}, seed {run.seed}",
    )
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, Path(path))


def detection_table(runs: list[SeedRun]) -> dict[str, tuple[float, float]]:
    """Taxa de detecção por categoria: (média, desvio) entre as seeds."""
    per_seed = [
        detection_rate_per_attack(r.errors, r.attacks, r.threshold, r.attack_names)
        for r in runs
    ]
    names = [n for n in runs[0].attack_names if n in per_seed[0]]
    return {
        n: (
            float(np.mean([p[n] for p in per_seed])),
            float(np.std([p[n] for p in per_seed])),
        )
        for n in names
    }


def detection_rates(runs: list[SeedRun], path: str | Path) -> Path:
    """Barras horizontais com a taxa de detecção de cada categoria.

    A barra da classe benigna é a taxa de falso positivo e fica em cinza.
    """
    table = detection_table(runs)
    names = sorted(table, key=lambda n: table[n][0])
    means = [table[n][0] for n in names]
    stds = [table[n][1] for n in names]
    colors = ["#a6a6a6" if n.lower() == "benign" else "#1f4e79" for n in names]
    fig, ax = plt.subplots(figsize=(5.5, 0.32 * len(names) + 1.2))
    ax.barh(names, means, xerr=stds, color=colors, capsize=2)
    for y, m in enumerate(means):
        ax.text(min(m + 0.02, 0.9), y, f"{m:.3f}", va="center", fontsize=7)
    ax.set(
        xlim=(0, 1.08),
        xlabel="Fração marcada como ataque (Benign = falso positivo)",
        title=f"{runs[0].dataset} ({len(runs)} seeds)",
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, Path(path))


def training_curves(runs: list[SeedRun], path: str | Path) -> Path:
    """Perda de treino/validação (log) e PR-AUC de validação por época."""
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(8.5, 3), layout="constrained")
    for i, run in enumerate(runs):
        if not run.history:
            continue
        c = SEED_COLORS[i % len(SEED_COLORS)]
        ep = [h["epoch"] for h in run.history]
        ax_loss.plot(ep, [h["train_loss"] for h in run.history], color=c, lw=1)
        ax_loss.plot(ep, [h["val_loss"] for h in run.history], color=c, lw=1, ls="--")
        ax_auc.plot(
            ep,
            [h["val_pr_auc"] for h in run.history],
            color=c,
            lw=1,
            label=f"seed {run.seed}",
        )
    ax_loss.set(yscale="log", xlabel="Época", ylabel="Perda (— treino, -- val.)")
    ax_auc.set(xlabel="Época", ylabel="PR-AUC de validação")
    for ax in (ax_loss, ax_auc):
        ax.grid(alpha=0.3)
    ax_auc.legend(fontsize=8)
    fig.suptitle(runs[0].dataset)
    return _save(fig, Path(path))


def all_figures(dataset_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Gera todas as figuras de um dataset em ``out_dir`` (PDF para o LaTeX)."""
    runs = load_runs(dataset_dir)
    if not runs:
        return []
    name = runs[0].dataset
    out = Path(out_dir)
    paths = [
        pr_curves(runs, out / f"{name}_pr.pdf"),
        detection_rates(runs, out / f"{name}_deteccao.pdf"),
        training_curves(runs, out / f"{name}_treino.pdf"),
    ]
    # Distribuição de scores: a seed mediana em PR-AUC representa o dataset.
    aps = [pr_auc(r.labels, r.errors) for r in runs]
    median_run = runs[int(np.argsort(aps)[len(aps) // 2])]
    paths.append(score_distribution(median_run, out / f"{name}_scores.pdf"))
    return paths
