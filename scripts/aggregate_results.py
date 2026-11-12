"""Agrega os resultados das execuções nas tabelas do relatório (entrega 3.4 da F3).

Lê todos os ``results/runs/<dataset>/seed<s>/metrics.json`` (contrato C2) e grava:

- ``tabela_reproducao.csv`` / ``.tex``: média ± desvio de PR-AUC e F1 macro no
  teste, lado a lado com a Tabela 3 do artigo, com a diferença e se a nossa
  média ficou a até 2 desvios do valor publicado (critério do PLANO.md, seção 2);
- ``tabela_custo.csv`` / ``.tex``: parâmetros, tempo de treino, inferência por
  amostra e pico de memória da GPU (para comparar com as Tabelas 2 e 4).

Uso:
    python scripts/aggregate_results.py                       # padrões abaixo
    python scripts/aggregate_results.py --runs results/runs --out results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

# Tabela 3 do artigo: (média, desvio) entre seeds, no conjunto de teste.
PAPER_TABLE3 = {
    "NF-UNSW-NB15-v3": {"pr_auc": (0.9998, 0.0007), "f1_macro": (0.9961, 0.0084)},
    "NF-CSE-CIC-IDS2018-v3": {
        "pr_auc": (0.8819, 0.0347),
        "f1_macro": (0.9447, 0.0213),
    },
    "NF-UNSW-NB15-v2": {"pr_auc": (0.8116, 0.0367), "f1_macro": (0.9264, 0.0217)},
    "NF-CSE-CIC-IDS2018-v2": {
        "pr_auc": (0.9201, 0.0238),
        "f1_macro": (0.9431, 0.0131),
    },
}

# Ordem das linhas nas tabelas: a da Tabela 3, depois os outros datasets.
DATASET_ORDER = list(PAPER_TABLE3)

# Métricas de qualidade: chave no metrics.json -> chave em PAPER_TABLE3.
QUALITY_METRICS = {"test_pr_auc": "pr_auc", "test_f1_macro": "f1_macro"}

# Métricas de custo: chave no metrics.json -> nome da coluna.
COST_METRICS = {
    "num_params": "Parâmetros",
    "train_time_s": "Treino (min)",
    "inference_us_per_sample": "Inferência (µs/amostra)",
    "peak_gpu_mem_mb": "Pico GPU (MB)",
}


def load_runs(runs_dir: str | Path) -> pd.DataFrame:
    """Lê todos os ``metrics.json`` abaixo de ``runs_dir``.

    Returns:
        DataFrame com uma linha por execução e uma coluna por chave do
        ``metrics.json``. Vazio se não houver nenhuma execução.
    """
    rows = []
    for path in sorted(Path(runs_dir).glob("*/seed*/metrics.json")):
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["path"] = str(path)
        rows.append(metrics)
    return pd.DataFrame(rows)


def _ordered(datasets) -> list[str]:
    """Datasets na ordem da Tabela 3, seguidos dos demais em ordem alfabética."""
    known = [d for d in DATASET_ORDER if d in set(datasets)]
    return known + sorted(set(datasets) - set(DATASET_ORDER))


def _mean_std(values: pd.Series, ddof: int) -> tuple[float, float]:
    """Média e desvio. Com uma só seed o desvio fica 0 (não há variação)."""
    std = float(values.std(ddof=ddof)) if len(values) > 1 else 0.0
    return float(values.mean()), std


def reproduction_table(runs: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """Tabela de reprodução: nossa média ± desvio contra a Tabela 3 do artigo.

    Para cada métrica ``m`` (``pr_auc`` e ``f1_macro``) as colunas são
    ``m_mean``, ``m_std``, ``m_paper_mean``, ``m_paper_std``, ``m_diff``
    (nosso − artigo) e ``m_within_2std`` (``|diff| <= 2 * desvio do artigo``).
    Datasets fora da Tabela 3 (ex.: NF-ToN-IoT-v3) ficam com NaN nas colunas
    do artigo.

    Args:
        runs: saída de ``load_runs``.
        ddof: graus de liberdade do desvio (1 = desvio amostral).
    """
    rows = []
    for dataset in _ordered(runs["dataset"].unique()):
        group = runs[runs["dataset"] == dataset]
        row = {
            "dataset": dataset,
            "n_seeds": len(group),
            "seeds": " ".join(str(s) for s in sorted(group["seed"])),
        }
        paper = PAPER_TABLE3.get(dataset, {})
        for key, name in QUALITY_METRICS.items():
            mean, std = _mean_std(group[key], ddof)
            p_mean, p_std = paper.get(name, (math.nan, math.nan))
            diff = mean - p_mean
            row |= {
                f"{name}_mean": mean,
                f"{name}_std": std,
                f"{name}_paper_mean": p_mean,
                f"{name}_paper_std": p_std,
                f"{name}_diff": diff,
                f"{name}_within_2std": (
                    bool(abs(diff) <= 2 * p_std) if name in paper else None
                ),
            }
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset")


def cost_table(runs: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """Tabela de custo: média ± desvio de cada métrica de ``COST_METRICS``.

    O tempo de treino sai em minutos, na coluna ``train_time_min``. Colunas:
    ``<chave>_mean`` e ``<chave>_std``.
    """
    rows = []
    for dataset in _ordered(runs["dataset"].unique()):
        group = runs[runs["dataset"] == dataset]
        row = {"dataset": dataset, "n_seeds": len(group)}
        for key in COST_METRICS:
            values, col = group[key], key
            if key == "train_time_s":
                values, col = values / 60, "train_time_min"
            row[f"{col}_mean"], row[f"{col}_std"] = _mean_std(values, ddof)
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset")


# --- Formatação LaTeX ------------------------------------------------------------


def _num(value: float, digits: int) -> str:
    """Número com vírgula decimal (padrão do relatório em português)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def _pm(mean: float, std: float, digits: int) -> str:
    if isinstance(mean, float) and math.isnan(mean):
        return "—"
    return f"${_num(mean, digits)} \\pm {_num(std, digits)}$"


def _latex_table(header, rows, caption, label, cols) -> str:
    """Tabela ``booktabs`` simples (precisa de ``\\usepackage{booktabs}``)."""
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        *(" & ".join(r) + " \\\\" for r in rows),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def reproduction_latex(table: pd.DataFrame) -> str:
    header = [
        "Dataset",
        "Seeds",
        "PR-AUC (nosso)",
        "PR-AUC (artigo)",
        "$\\Delta$",
        "F1 macro (nosso)",
        "F1 macro (artigo)",
        "$\\Delta$",
    ]
    rows = []
    for dataset, r in table.iterrows():
        cells = [dataset, str(r["n_seeds"])]
        for name in QUALITY_METRICS.values():
            cells += [
                _pm(r[f"{name}_mean"], r[f"{name}_std"], 4),
                _pm(r[f"{name}_paper_mean"], r[f"{name}_paper_std"], 4),
                _num(r[f"{name}_diff"], 4),
            ]
        rows.append(cells)
    return _latex_table(
        header,
        rows,
        "Reprodução do GraphIDS: média $\\pm$ desvio no teste entre seeds, "
        "comparada com a Tabela 3 do artigo.",
        "tab:reproducao",
        "lr" + "r" * 6,
    )


def cost_latex(table: pd.DataFrame) -> str:
    digits = {
        "num_params": 0,
        "train_time_min": 1,
        "inference_us_per_sample": 2,
        "peak_gpu_mem_mb": 0,
    }
    # µ como comando LaTeX, para compilar também com pdflatex.
    header = ["Dataset", *(c.replace("µ", "$\\mu$") for c in COST_METRICS.values())]
    rows = []
    for dataset, r in table.iterrows():
        cells = [dataset]
        for key, d in digits.items():
            if key == "num_params":  # igual em todas as seeds
                cells.append(f"{int(r[f'{key}_mean']):,}".replace(",", "."))
            else:
                cells.append(_pm(r[f"{key}_mean"], r[f"{key}_std"], d))
        rows.append(cells)
    return _latex_table(
        header,
        rows,
        "Custo computacional do GraphIDS por dataset (média $\\pm$ desvio entre "
        "seeds).",
        "tab:custo",
        "l" + "r" * len(COST_METRICS),
    )


def main(argv=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = argparse.ArgumentParser(description="Agrega os metrics.json (F3).")
    p.add_argument("--runs", default="results/runs", help="pasta das execuções")
    p.add_argument("--out", default="results", help="onde gravar as tabelas")
    p.add_argument(
        "--ddof", type=int, default=1, help="1 = desvio amostral, 0 = populacional"
    )
    args = p.parse_args(argv)

    runs = load_runs(args.runs)
    if runs.empty:
        raise SystemExit(f"Nenhum metrics.json encontrado em {args.runs}/*/seed*/")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repro = reproduction_table(runs, args.ddof)
    cost = cost_table(runs, args.ddof)
    repro.to_csv(out / "tabela_reproducao.csv")
    cost.to_csv(out / "tabela_custo.csv")
    (out / "tabela_reproducao.tex").write_text(
        reproduction_latex(repro), encoding="utf-8"
    )
    (out / "tabela_custo.tex").write_text(cost_latex(cost), encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = [
            "n_seeds",
            "pr_auc_mean",
            "pr_auc_paper_mean",
            "pr_auc_within_2std",
            "f1_macro_mean",
            "f1_macro_paper_mean",
            "f1_macro_within_2std",
        ]
        print(repro[cols].round(4))
    print(f"\nTabelas gravadas em {out}/tabela_{{reproducao,custo}}.{{csv,tex}}")
    return repro, cost


if __name__ == "__main__":
    main()
