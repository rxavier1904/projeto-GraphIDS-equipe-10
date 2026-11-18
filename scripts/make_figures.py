"""Gera as figuras e a tabela de detecção por ataque de todos os datasets.

Lê ``results/runs/<dataset>/seed*/`` (contrato C2) e grava em ``results/figuras/``:

- ``<dataset>_pr.pdf``, ``_scores.pdf``, ``_deteccao.pdf``, ``_treino.pdf``;
- ``deteccao_por_ataque.csv`` / ``.tex``: taxa de detecção média ± desvio de cada
  categoria de ataque em cada dataset (Benign = taxa de falso positivo).

Uso:
    python scripts/make_figures.py
    python scripts/make_figures.py --runs results/runs --out results/figuras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from graphids.evaluation.plots import all_figures, detection_table, load_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--out", default="results/figuras")
    args = parser.parse_args()

    rows = []
    for dataset_dir in sorted(p for p in Path(args.runs).iterdir() if p.is_dir()):
        paths = all_figures(dataset_dir, args.out)
        if not paths:
            print(f"{dataset_dir.name}: sem execuções completas, pulando")
            continue
        print(f"{dataset_dir.name}: {', '.join(p.name for p in paths)}")
        runs = load_runs(dataset_dir)
        for attack, (mean, std) in detection_table(runs).items():
            idx = runs[0].attack_names.index(attack)
            n = int(sum((r.attacks == idx).sum() for r in runs))
            rows.append(
                {
                    "dataset": runs[0].dataset,
                    "categoria": attack,
                    "fluxos_teste": n // len(runs),
                    "deteccao_media": mean,
                    "deteccao_desvio": std,
                    "n_seeds": len(runs),
                }
            )

    if not rows:
        return
    table = pd.DataFrame(rows)
    out = Path(args.out)
    table.to_csv(out / "deteccao_por_ataque.csv", index=False)
    tex = table.assign(
        deteccao=lambda d: (
            d.deteccao_media.map("{:.3f}".format)
            + " $\\pm$ "
            + d.deteccao_desvio.map("{:.3f}".format)
        )
    )[["dataset", "categoria", "fluxos_teste", "deteccao"]]
    tex.to_latex(out / "deteccao_por_ataque.tex", index=False, escape=False)
    print(f"Tabela gravada em {out / 'deteccao_por_ataque.csv'} e .tex")


if __name__ == "__main__":
    main()
