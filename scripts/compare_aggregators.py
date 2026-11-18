"""Compara visualmente os experimentos de ablação de agregadores no ToN-IoT.

Lê ``metrics.json``, ``history.json`` e ``scores_test.npz`` de cada pasta em
``results/runs/``:
- NF-ToN-IoT-v3          (Mean - Baseline)
- NF-ToN-IoT-v3-max      (Max)
- NF-ToN-IoT-v3-mean_max (Mean + Max)
- NF-ToN-IoT-v3-rich     (Rich: Mean + Max + Log-grau)

Suporta automaticamente agregação de múltiplas seeds (ex: 3 seeds) com
cálculo de média ± desvio padrão e barras de erro nos gráficos.

Gera:
1. Tabela resumo geral no terminal (PR-AUC, F1, Época, Tempo, GPU).
2. Tabela de detecção por tipo de ataque no terminal.
3. Figura com 4 gráficos gerais (PR-AUC Val/Test, Loss, F1).
4. Figura com gráfico de barras agrupadas de detecção por ataque.

Uso:
    python scripts/compare_aggregators.py                     # agrega todas as seeds
    python scripts/compare_aggregators.py --seed 1            # só seed 1
    python scripts/compare_aggregators.py --runs results/runs --out results/figuras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Adiciona a raiz do repositório ao sys.path para importar graphids
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from graphids.evaluation.metrics import detection_rate_per_attack  # noqa: E402

MODELS = [
    ("NF-ToN-IoT-v3", "Mean (Baseline)", "#1f77b4"),
    ("NF-ToN-IoT-v3-max", "Max", "#ff7f0e"),
    ("NF-ToN-IoT-v3-mean_max", "Mean + Max", "#2ca02c"),
    ("NF-ToN-IoT-v3-rich", "Rich (Mean+Max+LogDeg)", "#d62728"),
]


def load_single_seed(seed_dir: Path):
    metrics_file = seed_dir / "metrics.json"
    history_file = seed_dir / "history.json"
    scores_file = seed_dir / "scores_test.npz"

    if not metrics_file.exists():
        return None

    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)

    history = []
    if history_file.exists():
        with open(history_file, encoding="utf-8") as f:
            history = json.load(f)

    detection_rates = {}
    if scores_file.exists():
        try:
            npz = np.load(scores_file, allow_pickle=True)
            threshold = float(metrics.get("threshold", 0.0))
            errors = npz["errors"]
            attacks = npz["attacks"]
            attack_names = [str(n) for n in npz["attack_names"]]
            detection_rates = detection_rate_per_attack(
                errors, attacks, threshold, attack_names
            )
        except Exception as e:
            print(f"[Aviso] Erro ao ler {scores_file}: {e}")
    return {
        "metrics": metrics,
        "history": history,
        "detection_rates": detection_rates,
    }


def _calc_stats(vals):
    arr = np.array(vals, dtype=float)
    mean = float(np.mean(arr)) if len(arr) > 0 else 0.0
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _rates_from_csv(runs_dir: Path, dataset: str) -> dict:
    """(média, desvio) da detecção por categoria em ``deteccao_por_ataque.csv``."""
    csv = runs_dir.parent / "figuras" / "deteccao_por_ataque.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    sub = df[df["dataset"] == dataset]
    return {
        row.categoria: (float(row.deteccao_media), float(row.deteccao_desvio))
        for row in sub.itertuples()
    }


def load_experiment_data(runs_dir: Path, target_seed: int | None = None):
    results = []
    for folder, label, color in MODELS:
        folder_path = runs_dir / folder
        if not folder_path.exists():
            continue

        if target_seed is not None:
            seed_dirs = [folder_path / f"seed{target_seed}"]
        else:
            seed_dirs = sorted(folder_path.glob("seed*"))

        runs = []
        for sdir in seed_dirs:
            data = load_single_seed(sdir)
            if data is not None:
                runs.append(data)

        if not runs:
            continue

        pr_aucs = [r["metrics"].get("test_pr_auc", 0.0) for r in runs]
        f1s = [r["metrics"].get("test_f1_macro", 0.0) for r in runs]
        val_pr_aucs = [r["metrics"].get("val_best_pr_auc", 0.0) for r in runs]
        train_times = [r["metrics"].get("train_time_s", 0.0) for r in runs]
        gpus = [r["metrics"].get("peak_gpu_mem_mb", 0.0) for r in runs]
        best_eps = [r["metrics"].get("best_epoch", 1) for r in runs]

        pr_mean, pr_std = _calc_stats(pr_aucs)
        f1_mean, f1_std = _calc_stats(f1s)
        val_pr_mean, val_pr_std = _calc_stats(val_pr_aucs)
        time_mean, time_std = _calc_stats(train_times)
        gpu_mean, gpu_std = _calc_stats(gpus)

        # Taxas de detecção de ataque agrupadas por categoria
        all_cats = set()
        for r in runs:
            all_cats.update(r["detection_rates"].keys())

        agg_detection_rates = {}
        if not all_cats and target_seed is None:
            # Sem scores_test.npz locais (ficam fora do git): usa a média e o
            # desvio por seed já calculados em deteccao_por_ataque.csv.
            agg_detection_rates = _rates_from_csv(runs_dir, folder)
        for cat in all_cats:
            cat_vals = [
                r["detection_rates"][cat] for r in runs if cat in r["detection_rates"]
            ]
            if cat_vals:
                m_cat, s_cat = _calc_stats(cat_vals)
                agg_detection_rates[cat] = (m_cat, s_cat)

        # Média das curvas de validação
        epoch_map = {}
        for r in runs:
            for h in r["history"]:
                ep = h["epoch"]
                if ep not in epoch_map:
                    epoch_map[ep] = {"val_pr_auc": [], "val_loss": []}
                epoch_map[ep]["val_pr_auc"].append(h["val_pr_auc"])
                epoch_map[ep]["val_loss"].append(h["val_loss"])

        mean_history = []
        for ep in sorted(epoch_map):
            auc_m, auc_s = _calc_stats(epoch_map[ep]["val_pr_auc"])
            loss_m, loss_s = _calc_stats(epoch_map[ep]["val_loss"])
            mean_history.append(
                {
                    "epoch": ep,
                    "val_pr_auc": auc_m,
                    "val_pr_auc_std": auc_s,
                    "val_loss": loss_m,
                    "val_loss_std": loss_s,
                }
            )

        results.append(
            {
                "folder": folder,
                "label": label,
                "color": color,
                "n_seeds": len(runs),
                "pr_mean": pr_mean,
                "pr_std": pr_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "val_pr_mean": val_pr_mean,
                "val_pr_std": val_pr_std,
                "time_mean": time_mean,
                "time_std": time_std,
                "gpu_mean": gpu_mean,
                "gpu_std": gpu_std,
                "best_epoch": int(np.round(np.mean(best_eps))),
                "detection_rates": agg_detection_rates,
                "mean_history": mean_history,
            }
        )
    return results


def plot_comparison(data, out_path: Path, title_suffix: str = ""):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Comparativo de Agregadores do E-GraphSAGE (NF-ToN-IoT-v3{title_suffix})",
        fontsize=14,
        fontweight="bold",
    )

    # 1. PR-AUC de Validação por Época
    ax_prauc = axes[0, 0]
    for d in data:
        hist = d["mean_history"]
        if hist:
            epochs = [h["epoch"] for h in hist]
            vals = [h["val_pr_auc"] for h in hist]
            stds = [h["val_pr_auc_std"] for h in hist]
            best_ep = d["best_epoch"]
            ax_prauc.plot(
                epochs,
                vals,
                marker="o",
                markersize=4,
                label=d["label"],
                color=d["color"],
                linewidth=1.8,
            )
            if any(s > 0 for s in stds):
                vals_arr = np.array(vals)
                stds_arr = np.array(stds)
                ax_prauc.fill_between(
                    epochs,
                    vals_arr - stds_arr,
                    vals_arr + stds_arr,
                    color=d["color"],
                    alpha=0.15,
                )
            if best_ep <= len(vals):
                ax_prauc.scatter(
                    best_ep,
                    vals[best_ep - 1],
                    color=d["color"],
                    s=80,
                    edgecolors="black",
                    zorder=5,
                )
    ax_prauc.set_title("PR-AUC de Validação por Época")
    ax_prauc.set_xlabel("Época")
    ax_prauc.set_ylabel("PR-AUC")
    ax_prauc.grid(True, linestyle="--", alpha=0.6)
    ax_prauc.legend(frameon=True, fontsize=9)

    # 2. Perda de Validação por Época
    ax_loss = axes[0, 1]
    for d in data:
        hist = d["mean_history"]
        if hist:
            epochs = [h["epoch"] for h in hist]
            losses = [h["val_loss"] for h in hist]
            ax_loss.plot(
                epochs,
                losses,
                marker="s",
                markersize=3,
                label=d["label"],
                color=d["color"],
                linewidth=1.8,
            )
    ax_loss.set_title("Perda de Validação (MSE) por Época")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("MSE")
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend(frameon=True, fontsize=9)

    # 3. Barras de PR-AUC no Teste
    ax_bar_pr = axes[1, 0]
    labels = [d["label"] for d in data]
    pr_test = [d["pr_mean"] for d in data]
    pr_err = [d["pr_std"] for d in data]
    colors = [d["color"] for d in data]

    bars1 = ax_bar_pr.bar(
        labels,
        pr_test,
        yerr=pr_err if any(e > 0 for e in pr_err) else None,
        capsize=4,
        color=colors,
        width=0.55,
        edgecolor="black",
    )
    ax_bar_pr.set_title("PR-AUC no Conjunto de Teste (Média ± Desvio)")
    ax_bar_pr.set_ylabel("PR-AUC")
    max_val = max(pr_test, default=1.0)
    ax_bar_pr.set_ylim(0, min(1.0, max_val * 1.25))
    ax_bar_pr.grid(True, axis="y", linestyle="--", alpha=0.6)
    for i, bar in enumerate(bars1):
        yval = bar.get_height()
        err = pr_err[i]
        text = f"{yval:.4f}" if err == 0 else f"{yval:.4f}\n±{err:.4f}"
        ax_bar_pr.text(
            bar.get_x() + bar.get_width() / 2,
            yval + err + 0.015,
            text,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # 4. Barras de F1-Macro no Teste
    ax_bar_f1 = axes[1, 1]
    f1_test = [d["f1_mean"] for d in data]
    f1_err = [d["f1_std"] for d in data]

    bars2 = ax_bar_f1.bar(
        labels,
        f1_test,
        yerr=f1_err if any(e > 0 for e in f1_err) else None,
        capsize=4,
        color=colors,
        width=0.55,
        edgecolor="black",
    )
    ax_bar_f1.set_title("F1-Macro no Conjunto de Teste (Média ± Desvio)")
    ax_bar_f1.set_ylabel("F1-Macro")
    max_f1 = max(f1_test, default=1.0)
    ax_bar_f1.set_ylim(0, min(1.0, max_f1 * 1.25))
    ax_bar_f1.grid(True, axis="y", linestyle="--", alpha=0.6)
    for i, bar in enumerate(bars2):
        yval = bar.get_height()
        err = f1_err[i]
        text = f"{yval:.4f}" if err == 0 else f"{yval:.4f}\n±{err:.4f}"
        ax_bar_f1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + err + 0.015,
            text,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    out_png = out_path / "comparativo_agregadores.png"
    out_pdf = out_path / "comparativo_agregadores.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Gráfico comparativo gravado em:\n  - {out_png}\n  - {out_pdf}")


def plot_attacks_bar(data, out_path: Path):
    has_rates = [d for d in data if d.get("detection_rates")]
    if not has_rates:
        return

    all_cats = []
    for d in has_rates:
        for cat in d["detection_rates"]:
            if cat not in all_cats:
                all_cats.append(cat)

    attacks = sorted([c for c in all_cats if c.lower() != "benign"])
    if not attacks:
        return

    fig, ax = plt.subplots(figsize=(15, 7.5))
    n_models = len(has_rates)
    width = 0.8 / n_models
    x = np.arange(len(attacks))

    for i, d in enumerate(has_rates):
        rates = [d["detection_rates"].get(atk, (0.0, 0.0))[0] * 100 for atk in attacks]
        errs = [d["detection_rates"].get(atk, (0.0, 0.0))[1] * 100 for atk in attacks]
        has_err = any(e > 0 for e in errs)
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            rates,
            width,
            yerr=errs if has_err else None,
            capsize=3,
            label=f"{d['label']} ({d['n_seeds']} seeds)",
            color=d["color"],
            edgecolor="black",
            linewidth=0.8,
        )
        for j, b in enumerate(bars):
            h = b.get_height()
            err_val = errs[j] if has_err else 0.0
            if h > 2:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + err_val + 1,
                    f"{h:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(
        "Taxa de Detecção por Categoria de Ataque (%) - NF-ToN-IoT-v3",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Taxa de Detecção (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 118)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    out_png = out_path / "comparativo_ataques.png"
    out_pdf = out_path / "comparativo_ataques.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico por ataque gravado em:\n  - {out_png}\n  - {out_pdf}")


def print_attack_detection_table(data):
    has_rates = [d for d in data if d.get("detection_rates")]
    if not has_rates:
        print("\n[Aviso] Nenhum scores_test.npz encontrado para taxas por ataque.")
        return

    all_cats = []
    for d in has_rates:
        for cat in d["detection_rates"]:
            if cat not in all_cats:
                all_cats.append(cat)

    attacks = sorted([c for c in all_cats if c.lower() != "benign"])
    if any(c.lower() == "benign" for c in all_cats):
        attacks.append("Benign (Taxa Falso Positivo)")

    rows = []
    for cat in attacks:
        key = "Benign" if "Benign" in cat else cat
        row = {"Categoria de Ataque": cat}
        for d in has_rates:
            rates = d["detection_rates"]
            if key in rates:
                m, s = rates[key]
                val_m = m * 100
                val_s = s * 100
                if d["n_seeds"] > 1 and val_s > 0:
                    row[d["label"]] = f"{val_m:5.1f}% ± {val_s:.1f}%"
                else:
                    row[d["label"]] = f"{val_m:5.1f}%"
            else:
                row[d["label"]] = "  N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(" TAXA DE DETECÇÃO POR CATEGORIA DE ATAQUE (TESTE - MÉDIA ± DESVIO)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


def _fmt_stat(m: float, s: float, n: int, digits: int = 4) -> str:
    if n > 1 and s > 0:
        return f"{m:.{digits}f} ± {s:.{digits}f}"
    return f"{m:.{digits}f}"


def print_summary_table(data):
    rows = []
    for d in data:
        n = d["n_seeds"]
        train_min = d["time_mean"] / 60
        train_std_min = d["time_std"] / 60

        rows.append(
            {
                "Agregador": d["label"],
                "Seeds": n,
                "PR-AUC (Teste)": _fmt_stat(d["pr_mean"], d["pr_std"], n),
                "F1-Macro (Teste)": _fmt_stat(d["f1_mean"], d["f1_std"], n),
                "PR-AUC (Val)": _fmt_stat(d["val_pr_mean"], d["val_pr_std"], n),
                "Melhor Época": d["best_epoch"],
                "Treino (min)": _fmt_stat(train_min, train_std_min, n, digits=1),
                "GPU Pico (MB)": _fmt_stat(d["gpu_mean"], d["gpu_std"], n, digits=1),
            }
        )
    df = pd.DataFrame(rows)
    print("\n" + "=" * 95)
    print(" RESUMO COMPARATIVO GERAL DE AGREGADORES (NF-ToN-IoT-v3 - MÉDIA ± DESVIO)")
    print("=" * 95)
    print(df.to_string(index=False))
    print("=" * 95)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="results/runs", help="diretório de runs")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed específica (padrão: agrega todas as seeds disponíveis)",
    )
    p.add_argument(
        "--out", default="results/figuras", help="diretório de saída dos gráficos"
    )
    args = p.parse_args()

    runs_dir = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data(runs_dir, args.seed)
    if not data:
        msg = f" para a seed {args.seed}" if args.seed is not None else ""
        print(f"Nenhum resultado encontrado em {runs_dir}{msg}.")
        return

    suffix = f" - Seed {args.seed}" if args.seed is not None else " - Média 3 Seeds"
    print_summary_table(data)
    print_attack_detection_table(data)
    plot_comparison(data, out_dir, title_suffix=suffix)
    plot_attacks_bar(data, out_dir)


if __name__ == "__main__":
    main()
