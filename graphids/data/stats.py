"""Estatísticas dos datasets para a seção 6 do relatório (entrega 2.5 da F2).

Gera duas tabelas por dataset:

1. **Resumo** do CSV completo: número de fluxos, de hosts e proporção de
   ataques, lado a lado com a Tabela 1 do artigo.
2. **Contagem por split**: fluxos de cada categoria de ataque no treino, na
   validação e no teste, exatamente como o modelo os recebe (lidos com
   ``load_splits``, que usa o cache). A especificação do projeto pede a
   quantidade de dados por classe em cada split.

Cada tabela é salva em CSV e em LaTeX (para colar no Overleaf).

Uso pela linha de comando (uma ou mais configs):

    python -m graphids.data.stats --config configs/NF-UNSW-NB15-v3.yaml
    python -m graphids.data.stats --config configs/NF-UNSW-NB15-v2.yaml \\
        --config configs/NF-UNSW-NB15-v3.yaml --data-dir data --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from graphids.config import load_config
from graphids.data.loader import load_splits
from graphids.data.loading import (
    ATTACK_COL,
    BENIGN,
    DST_IP_COL,
    LABEL_COL,
    SRC_IP_COL,
    FlowTable,
    load_flow_table,
)
from graphids.data.types import DataConfig, GraphSplits

# Tabela 1 do artigo: (fluxos, hosts, proporção de ataques em %).
PAPER_TABLE1 = {
    "NF-UNSW-NB15-v2": (2_390_275, 44, 3.98),
    "NF-UNSW-NB15-v3": (2_365_424, 44, 5.40),
    "NF-CSE-CIC-IDS2018-v2": (18_893_708, 255_042, 11.95),
    "NF-CSE-CIC-IDS2018-v3": (20_115_529, 205_801, 12.93),
}
SPLITS = ("train", "val", "test")
SPLIT_LABELS = {"train": "Treino", "val": "Validação", "test": "Teste"}


# --- Cálculo das tabelas ------------------------------------------------------


def dataset_summary(table: FlowTable, dataset: str) -> pd.Series:
    """Fluxos, hosts e proporção de ataques do CSV inteiro (antes do split).

    Hosts = IPs distintos que aparecem como origem ou destino. Quando o dataset
    está na Tabela 1 do artigo, os valores publicados vêm ao lado.
    """
    flows = table.flows
    src_ips = flows[SRC_IP_COL].cat.categories
    hosts = src_ips.union(flows[DST_IP_COL].cat.categories)
    categories = flows[ATTACK_COL].cat.categories
    # Cada valor nosso fica ao lado do valor publicado ("—" fora da Tabela 1).
    paper = PAPER_TABLE1.get(dataset, (None, None, None))
    summary = {
        "Fluxos": len(flows),
        "Fluxos (artigo)": paper[0],
        "Hosts": len(hosts),
        "Hosts (artigo)": paper[1],
        "Ataques (%)": round(100 * float(flows[LABEL_COL].mean()), 2),
        "Ataques % (artigo)": paper[2],
        "Categorias de ataque": int((categories != BENIGN).sum()),
    }
    # dtype=object mantém os inteiros como inteiros na tabela final.
    return pd.Series(summary, name=dataset, dtype=object)


def split_class_counts(splits: GraphSplits) -> pd.DataFrame:
    """Fluxos por categoria de ataque em cada split, com linhas e coluna de total.

    Linhas: ``Benign``, cada categoria de ataque, ``Ataques (total)`` e
    ``Total``. Colunas: ``Treino``, ``Validação``, ``Teste`` e ``Total``.
    O treino tem zero ataques, porque eles são removidos (Seção 4.2).
    """
    names = splits.train.attack_names
    columns = {}
    for split in SPLITS:
        attack = getattr(splits, split).edge_attack.cpu().numpy()
        columns[SPLIT_LABELS[split]] = np.bincount(attack, minlength=len(names))
    counts = pd.DataFrame(columns, index=names)
    counts["Total"] = counts.sum(axis=1)
    counts.loc["Ataques (total)"] = counts.iloc[1:].sum()
    counts.loc["Total"] = counts.loc[names].sum()
    counts.index.name = "Categoria"
    return counts.astype(int)


def split_hosts(splits: GraphSplits) -> pd.Series:
    """Número de hosts que aparecem em alguma aresta de cada split."""
    return pd.Series(
        {
            SPLIT_LABELS[s]: int(
                np.unique(getattr(splits, s).edge_index.cpu().numpy()).size
            )
            for s in SPLITS
        },
        name="Hosts com arestas",
    )


# --- Exportação ---------------------------------------------------------------


def format_number(value) -> str:
    """Formata números no padrão brasileiro: 1.790.185 e 5,40."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"  # ex.: dataset fora da Tabela 1 do artigo
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", ".")
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}".replace(".", ",")
    return str(value)


def _latex_escape(text: str) -> str:
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Tabela LaTeX simples (``booktabs``), sem depender do jinja2.

    Precisa de ``\\usepackage{booktabs}`` no preâmbulo do Overleaf.
    """
    cols = "l" + "r" * len(df.columns)
    index_name = _latex_escape(str(df.index.name or ""))
    header = " & ".join([index_name, *(_latex_escape(str(c)) for c in df.columns)])
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for idx, row in df.iterrows():
        cells = [_latex_escape(str(idx)), *(format_number(v) for v in row)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def export(df: pd.DataFrame, path_stem: Path, caption: str, label: str) -> None:
    """Grava ``<path_stem>.csv`` e ``<path_stem>.tex``."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_stem.with_suffix(".csv"))
    path_stem.with_suffix(".tex").write_text(to_latex(df, caption, label))


# --- Execução -----------------------------------------------------------------


def dataset_statistics(cfg: DataConfig, out_dir: str | Path) -> pd.Series:
    """Calcula e exporta as tabelas de um dataset. Devolve o resumo.

    Arquivos gerados em ``out_dir``:
        ``<dataset>_contagem_por_split.csv/.tex``: fluxos por categoria em
        cada split, mais uma linha com os hosts de cada split.
    """
    table = load_flow_table(cfg.data_dir, cfg.dataset, cfg.drop_timestamps)
    summary = dataset_summary(table, cfg.dataset)
    # Reaproveita a tabela já lida se ainda não houver cache (evita ler o
    # CSV duas vezes, o que no CIC-IDS2018 leva minutos).
    splits = load_splits(cfg, table=table)
    del table
    counts = split_class_counts(splits)
    # Última linha: hosts que aparecem em cada split. Na coluna Total, o
    # número de nós do grafo (mapa de IPs compartilhado entre os splits).
    hosts = split_hosts(splits)
    counts.loc["Hosts com arestas"] = [*hosts.to_numpy(), splits.train.num_nodes]
    sample = f", amostra de {cfg.fraction:g}" if cfg.fraction else ""
    export(
        counts,
        Path(out_dir) / f"{cfg.dataset}_contagem_por_split",
        caption=f"Fluxos por categoria e split no {cfg.dataset} "
        f"(seed {cfg.seed}{sample}). Os ataques que cairiam no treino são "
        "removidos, por isso o treino só tem fluxos benignos.",
        label=f"tab:split-{cfg.dataset.lower()}",
    )
    return summary


def main(argv=None) -> pd.DataFrame:
    """Gera as tabelas de cada config e um resumo comparando com o artigo."""
    p = argparse.ArgumentParser(description="Estatísticas dos datasets (F2).")
    p.add_argument("--config", action="append", required=True, help="YAML (repete)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results/estatisticas")
    args = p.parse_args(argv)

    summaries = []
    for path in args.config:
        cfg = load_config(path, [f"seed={args.seed}", f"data_dir={args.data_dir}"])
        print(f"Processando {cfg.dataset}...")
        summaries.append(dataset_statistics(cfg.data_config(), args.out))

    # dtype=object guarda cada número com o tipo original (inteiros continuam
    # inteiros na tabela LaTeX, e valores ausentes viram "—").
    resumo = pd.DataFrame(
        [s.to_dict() for s in summaries],
        index=[s.name for s in summaries],
        dtype=object,
    )
    resumo.index.name = "Dataset"
    export(
        resumo,
        Path(args.out) / "resumo_datasets",
        caption="Estatísticas dos datasets NetFlow, comparadas com a Tabela 1 "
        "do artigo.",
        label="tab:resumo-datasets",
    )
    print(resumo.T.to_string())
    print(f"Tabelas gravadas em {args.out}/")
    return resumo


if __name__ == "__main__":
    main()
