"""Tabela da ablação de agregação: baseline (mean) × variantes, seed 1.

Lê ``results/runs/<dataset>/seed1/`` (baseline) e
``results/runs_agg/<variante>/<dataset>/seed1/`` e grava
``results/tabela_agregacao.csv`` com PR-AUC, F1 macro, tempo de inferência,
tempo de treino e o recall de cada categoria de ataque no teste (com o
threshold escolhido na validação, o mesmo de ``metrics.json``).

Uso:
    python scripts/aggregate_agg.py [--runs results/runs] [--agg results/runs_agg]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def per_attack_recall(run: Path, threshold: float) -> dict[str, float]:
    path = run / "scores_test.npz"
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    preds = z["errors"] > threshold
    names = [str(n) for n in z["attack_names"]]
    out = {}
    for i, name in enumerate(names):
        if i == 0:
            continue  # Benign
        mask = z["attacks"] == i
        if mask.any():
            out[f"recall_{name}"] = float(preds[mask].mean())
    benign = z["attacks"] == 0
    out["fpr_benign"] = float(preds[benign].mean()) if benign.any() else float("nan")
    return out


def row(run: Path, agg: str) -> dict | None:
    f = run / "metrics.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    r = {
        "dataset": m["dataset"],
        "agg_type": agg,
        "test_pr_auc": m["test_pr_auc"],
        "test_f1_macro": m["test_f1_macro"],
        "inference_us_per_sample": m["inference_us_per_sample"],
        "train_time_s": m["train_time_s"],
        "best_epoch": m["best_epoch"],
        "num_params": m["num_params"],
    }
    r.update(per_attack_recall(run, m["threshold"]))
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--agg", default="results/runs_agg")
    p.add_argument("--out", default="results/tabela_agregacao.csv")
    args = p.parse_args()

    rows = []
    for ds in sorted(Path(args.runs).iterdir()):
        r = row(ds / "seed1", "mean")
        if r:
            rows.append(r)
    agg_root = Path(args.agg)
    if agg_root.exists():
        for variant in sorted(agg_root.iterdir()):
            for ds in sorted(variant.iterdir()):
                r = row(ds / "seed1", variant.name)
                if r:
                    rows.append(r)

    df = pd.DataFrame(rows).sort_values(["dataset", "agg_type"])
    df.to_csv(args.out, index=False)
    cols = [
        "dataset",
        "agg_type",
        "test_pr_auc",
        "test_f1_macro",
        "inference_us_per_sample",
        "train_time_s",
        "best_epoch",
    ]
    with pd.option_context(
        "display.width", 160, "display.float_format", "{:.4f}".format
    ):
        print(df[cols].to_string(index=False))
    print(f"\ngravado em {args.out}")


if __name__ == "__main__":
    main()
