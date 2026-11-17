"""Compara visualmente os experimentos de ablação de agregadores no ToN-IoT.

Lê ``metrics.json``, ``history.json`` e ``scores_test.npz`` de cada pasta em
``results/runs/``:
- NF-ToN-IoT-v3          (Mean - Baseline)
- NF-ToN-IoT-v3-max      (Max)
- NF-ToN-IoT-v3-mean_max (Mean + Max)
- NF-ToN-IoT-v3-rich     (Rich: Mean + Max + Log-grau)

Gera:
1. Tabela resumo geral no terminal (PR-AUC, F1, Época, Tempo, GPU).
2. Tabela de detecção por tipo de ataque no terminal (se scores_test.npz existir).
3. Figura com 4 gráficos gerais (PR-AUC Val/Test, Loss, F1).
4. Figura com gráfico de barras agrupadas de detecção por ataque.

Uso:
    python scripts/compare_aggregators.py
    python scripts/compare_aggregators.py --runs results/runs --seed 1 --out results
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


def load_experiment_data(runs_dir: Path, seed: int):
    results = []
    for folder, label, color in MODELS:
        seed_dir = runs_dir / folder / f"seed{seed}"
        metrics_file = seed_dir / "metrics.json"
        history_file = seed_dir / "history.json"
        scores_file = seed_dir / "scores_test.npz"

        if not metrics_file.exists():
            continue

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
        elif folder == "NF-ToN-IoT-v3":
            # Fallback para os resultados oficiais do baseline (deteccao_por_ataque.csv)
            for cand in [
                runs_dir.parent / "figuras" / "deteccao_por_ataque.csv",
                Path("results/figuras/deteccao_por_ataque.csv"),
            ]:
                if cand.exists():
                    try:
                        df_det = pd.read_csv(cand)
                        sub = df_det[df_det["dataset"] == "NF-ToN-IoT-v3"]
                        detection_rates = dict(
                            zip(sub["categoria"], sub["deteccao_media"], strict=False)
                        )
                        break
                    except Exception:
                        pass
            if not detection_rates:
                detection_rates = {
                    "Backdoor": 1.0,
                    "ddos": 0.49819,
                    "dos": 0.36373,
                    "injection": 0.67701,
                    "mitm": 0.34444,
                    "password": 0.43420,
                    "ransomware": 1.0,
                    "scanning": 0.04246,
                    "xss": 0.40475,
                    "Benign": 0.09904,
                }

        results.append(
            {
                "folder": folder,
                "label": label,
                "color": color,
                "metrics": metrics,
                "history": history,
                "detection_rates": detection_rates,
            }
        )
    return results


def plot_comparison(data, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Comparativo de Agregadores do E-GraphSAGE (NF-ToN-IoT-v3 - Seed 1)",
        fontsize=14,
        fontweight="bold",
    )

    # 1. PR-AUC de Validação por Época
    ax_prauc = axes[0, 0]
    for d in data:
        hist = d["history"]
        if hist:
            epochs = [h["epoch"] for h in hist]
            vals = [h["val_pr_auc"] for h in hist]
            best_ep = d["metrics"].get("best_epoch", 1)
            ax_prauc.plot(
                epochs,
                vals,
                marker="o",
                markersize=4,
                label=d["label"],
                color=d["color"],
                linewidth=1.8,
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
        hist = d["history"]
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
    pr_test = [d["metrics"].get("test_pr_auc", 0.0) for d in data]
    colors = [d["color"] for d in data]

    bars1 = ax_bar_pr.bar(labels, pr_test, color=colors, width=0.55, edgecolor="black")
    ax_bar_pr.set_title("PR-AUC no Conjunto de Teste")
    ax_bar_pr.set_ylabel("PR-AUC")
    ax_bar_pr.set_ylim(0, min(1.0, max(pr_test, default=1.0) * 1.15))
    ax_bar_pr.grid(True, axis="y", linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax_bar_pr.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Barras de F1-Macro no Teste
    ax_bar_f1 = axes[1, 1]
    f1_test = [d["metrics"].get("test_f1_macro", 0.0) for d in data]

    bars2 = ax_bar_f1.bar(labels, f1_test, color=colors, width=0.55, edgecolor="black")
    ax_bar_f1.set_title("F1-Macro no Conjunto de Teste")
    ax_bar_f1.set_ylabel("F1-Macro")
    ax_bar_f1.set_ylim(0, min(1.0, max(f1_test, default=1.0) * 1.15))
    ax_bar_f1.grid(True, axis="y", linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax_bar_f1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
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

    fig, ax = plt.subplots(figsize=(14, 7))
    n_models = len(has_rates)
    width = 0.8 / n_models
    x = np.arange(len(attacks))

    for i, d in enumerate(has_rates):
        rates = [d["detection_rates"].get(atk, 0.0) * 100 for atk in attacks]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            rates,
            width,
            label=d["label"],
            color=d["color"],
            edgecolor="black",
            linewidth=0.8,
        )
        for b in bars:
            h = b.get_height()
            if h > 2:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 1,
                    f"{h:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    ax.set_title(
        "Taxa de Detecção por Categoria de Ataque (%) - NF-ToN-IoT-v3 (Seed 1)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Taxa de Detecção (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 115)
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
                val = rates[key] * 100
                row[d["label"]] = f"{val:5.1f}%"
            else:
                row[d["label"]] = "  N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 85)
    print(" TAXA DE DETECÇÃO POR CATEGORIA DE ATAQUE (TESTE - SEED 1)")
    print("=" * 85)
    print(df.to_string(index=False))
    print("=" * 85 + "\n")


def print_summary_table(data):
    rows = []
    for d in data:
        m = d["metrics"]
        train_s = m.get("train_time_s", 0.0)
        rows.append(
            {
                "Agregador": d["label"],
                "PR-AUC (Teste)": f"{m.get('test_pr_auc', 0.0):.4f}",
                "F1-Macro (Teste)": f"{m.get('test_f1_macro', 0.0):.4f}",
                "PR-AUC (Val)": f"{m.get('val_best_pr_auc', 0.0):.4f}",
                "Melhor Época": m.get("best_epoch", "-"),
                "Treino (min)": f"{train_s / 60:.1f}",
                "GPU Pico (MB)": f"{m.get('peak_gpu_mem_mb', 0.0):.1f}",
            }
        )
    df = pd.DataFrame(rows)
    print("\n" + "=" * 85)
    print(" RESUMO COMPARATIVO GERAL DE AGREGADORES (NF-ToN-IoT-v3)")
    print("=" * 85)
    print(df.to_string(index=False))
    print("=" * 85)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="results/runs", help="diretório de runs")
    p.add_argument("--seed", type=int, default=1, help="seed a comparar")
    p.add_argument("--out", default="results", help="diretório de saída dos gráficos")
    args = p.parse_args()

    runs_dir = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data(runs_dir, args.seed)
    if not data:
        print(f"Nenhum resultado encontrado em {runs_dir} para a seed {args.seed}.")
        return

    print_summary_table(data)
    print_attack_detection_table(data)
    plot_comparison(data, out_dir)
    plot_attacks_bar(data, out_dir)


if __name__ == "__main__":
    main()
