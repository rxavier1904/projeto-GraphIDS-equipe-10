"""Estudo do threshold de decisão sobre os scores já salvos (sem retreinar).

O artigo escolhe o threshold que maximiza o F1 macro na VALIDAÇÃO, usando os
rótulos de ataque. Aqui comparamos com regras que não usam rótulos de ataque:

- ``f1_val``: o do artigo (usa rótulos de ataque da validação);
- ``p95``, ``p99``, ``p99.9``: percentil dos scores BENIGNOS da validação
  (só exige saber o que é tráfego normal, como no treino);
- ``mad3``, ``mad10``: mediana + k·MAD de TODOS os scores da validação
  (sem rótulo nenhum, ``graphids.training.threshold.mad_threshold``);
- ``oraculo``: o melhor threshold no próprio TESTE (limite superior, não é
  utilizável na prática).

Para cada execução em ``results/runs/<dataset>/seed<s>/`` grava F1 macro,
taxa de detecção (recall do ataque) e taxa de falso positivo no teste em
``results/tabela_threshold.csv`` e a média ± desvio entre seeds em
``results/tabela_threshold_resumo.csv``.

Uso:
    python scripts/threshold_study.py [--runs results/runs]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from graphids.training.threshold import best_f1_threshold, mad_threshold


def rules(val_scores, val_labels, test_scores, test_labels) -> dict[str, float]:
    benign = val_scores[val_labels == 0]
    out = {
        "f1_val": best_f1_threshold(val_scores, val_labels, 500)[0],
        "p95": float(np.percentile(benign, 95)),
        "p99": float(np.percentile(benign, 99)),
        "p99.9": float(np.percentile(benign, 99.9)),
        "mad3": mad_threshold(val_scores, 3.0),
        "mad10": mad_threshold(val_scores, 10.0),
    }
    out["oraculo"] = best_f1_threshold(test_scores, test_labels, 2000)[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    rows = []
    for run in sorted(Path(args.runs).glob("*/seed*")):
        if not (run / "scores_test.npz").exists():
            continue
        v, t = np.load(run / "scores_val.npz"), np.load(run / "scores_test.npz")
        ys, s = t["labels"], t["errors"]
        for rule, thr in rules(v["errors"], v["labels"], s, ys).items():
            pred = (s > thr).astype(int)
            rows.append(
                {
                    "dataset": run.parent.name,
                    "seed": int(run.name.removeprefix("seed")),
                    "regra": rule,
                    "f1_macro": f1_score(ys, pred, average="macro", zero_division=0),
                    "deteccao": pred[ys == 1].mean(),
                    "fpr": pred[ys == 0].mean(),
                }
            )

    df = pd.DataFrame(rows)
    out = Path(args.out)
    df.to_csv(out / "tabela_threshold.csv", index=False)
    order = ["f1_val", "p95", "p99", "p99.9", "mad3", "mad10", "oraculo"]
    summ = (
        df.groupby(["dataset", "regra"])[["f1_macro", "deteccao", "fpr"]]
        .agg(["mean", "std"])
        .reindex(order, level=1)
    )
    summ.to_csv(out / "tabela_threshold_resumo.csv")
    with pd.option_context(
        "display.width", 200, "display.float_format", "{:.3f}".format
    ):
        print(
            df.groupby(["dataset", "regra"])[["f1_macro", "deteccao", "fpr"]]
            .mean()
            .reindex(order, level=1)
            .to_string()
        )


if __name__ == "__main__":
    main()
