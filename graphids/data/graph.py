"""Construção dos grafos de treino, validação e teste.

Seção 3.1 do artigo: os nós são hosts (endereços IP) e cada fluxo vira uma
aresta dirigida do IP de origem para o IP de destino, com as features do fluxo
na aresta. O grafo é dirigido porque ataques têm direção (quem varre e quem é
varrido).

Como no código oficial, o mapa IP → índice de nó é COMPARTILHADO: ele é criado
com a união dos IPs dos três splits, de modo que o mesmo host tem o mesmo
índice no treino, na validação e no teste. Cada split continua sendo um grafo
separado: as arestas não se misturam (Seção 4.2).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from graphids.data.loading import (
    ATTACK_COL,
    BENIGN,
    DST_IP_COL,
    LABEL_COL,
    SRC_IP_COL,
    FlowTable,
)
from graphids.data.preprocessing import features_at, fit_scaler, transform_features
from graphids.data.split import SplitIndices
from graphids.data.types import FlowGraph, GraphSplits


@dataclass
class BuiltDataset:
    """Tudo o que o pipeline produz para um dataset.

    Atributos:
        splits: os três grafos (contrato C1).
        hosts: IP de cada índice de nó (``hosts[i]`` é o IP do nó ``i``).
            Permite ligar um alerta do modelo de volta ao host real.
        scaler: Min-Max ajustado no treino, salvo junto do cache.
    """

    splits: GraphSplits
    hosts: list[str]
    scaler: MinMaxScaler


def host_index(table: FlowTable, splits: SplitIndices):
    """Mapa IP → índice de nó, compartilhado entre os três splits.

    Usa só os fluxos que ficaram em algum split: hosts que aparecem apenas em
    ataques removidos do treino não viram nós.

    Returns:
        ``(src, dst, hosts)``: ``src`` e ``dst`` são arrays ``int64`` com o
        índice de nó de cada linha da tabela (``-1`` nas linhas fora de todos
        os splits); ``hosts`` lista o IP de cada índice.
    """
    used = np.concatenate([splits.train, splits.val, splits.test])
    src_col = table.flows[SRC_IP_COL]
    dst_col = table.flows[DST_IP_COL]
    for col in (src_col, dst_col):
        if (col.cat.codes.to_numpy()[used] < 0).any():
            raise ValueError(f"Há fluxos sem IP na coluna {col.name}.")

    # As colunas de IP são do tipo category: os códigos inteiros indicam qual
    # IP aparece em cada linha. Pegamos os IPs presentes nas linhas usadas.
    src_ips = src_col.cat.categories[np.unique(src_col.cat.codes.to_numpy()[used])]
    dst_ips = dst_col.cat.categories[np.unique(dst_col.cat.codes.to_numpy()[used])]
    hosts = src_ips.union(dst_ips)  # união ordenada dos IPs

    # Recodifica as duas colunas com a mesma lista de hosts. IPs fora da lista
    # (só em linhas não usadas) recebem o código -1.
    src = src_col.cat.set_categories(hosts).cat.codes.to_numpy().astype(np.int64)
    dst = dst_col.cat.set_categories(hosts).cat.codes.to_numpy().astype(np.int64)
    return src, dst, [str(h) for h in hosts]


def attack_index(table: FlowTable):
    """Índice da categoria de ataque de cada linha, com ``"Benign"`` na posição 0.

    O contrato C1 exige ``attack_names[0] == "Benign"``. As demais categorias
    ficam em ordem alfabética, iguais nos três splits.

    Returns:
        ``(codes, attack_names)``: ``codes`` é um array ``int64`` com um índice
        por linha da tabela.
    """
    col = table.flows[ATTACK_COL]
    present = [str(c) for c in col.cat.categories]
    if BENIGN not in present:
        raise ValueError(f"Nenhum fluxo {BENIGN!r} na coluna {ATTACK_COL}.")
    names = [BENIGN] + sorted(c for c in present if c != BENIGN)
    codes = col.cat.reorder_categories(names).cat.codes.to_numpy().astype(np.int64)
    return codes, names


def _check_labels(label: np.ndarray, attack: np.ndarray) -> None:
    """Avisa se ``Label`` e ``Attack`` discordam (deveria ser Label = Attack ≠ 0)."""
    disagree = int(((attack != 0).astype(np.int8) != label).sum())
    if disagree:
        warnings.warn(
            f"{disagree} fluxos com Label inconsistente com Attack; o modelo "
            "usa Label como rótulo.",
            stacklevel=3,
        )


def build_graphs(table: FlowTable, splits: SplitIndices) -> BuiltDataset:
    """Normaliza as features e monta os três ``FlowGraph`` (contrato C1).

    Args:
        table: fluxos limpos (``loading.py``), já amostrados se for o caso.
        splits: índices de cada split (``split.py``), com o treino só benigno.

    Returns:
        ``BuiltDataset`` com os grafos, a lista de hosts e o scaler.
    """
    src, dst, hosts = host_index(table, splits)
    attack, attack_names = attack_index(table)
    label = table.flows[LABEL_COL].to_numpy().astype(np.int64)
    _check_labels(label, attack)

    # 2.3: Min-Max ajustado SÓ no treino benigno, aplicado aos três splits.
    train_x = features_at(table, splits.train)
    scaler = fit_scaler(train_x)

    graphs = {}
    for name, rows in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        x = train_x if name == "train" else features_at(table, rows)
        x = transform_features(scaler, x)
        graphs[name] = FlowGraph(
            # [2, E]: linha 0 = host de origem, linha 1 = host de destino.
            edge_index=torch.as_tensor(np.stack([src[rows], dst[rows]])),
            edge_attr=torch.as_tensor(x),
            edge_label=torch.as_tensor(label[rows]),
            edge_attack=torch.as_tensor(attack[rows]),
            num_nodes=len(hosts),  # o mesmo nos três splits
            attack_names=list(attack_names),
        )

    graph_splits = GraphSplits(
        train=graphs["train"],
        val=graphs["val"],
        test=graphs["test"],
        feature_names=list(table.feature_names),
    )
    return BuiltDataset(splits=graph_splits, hosts=hosts, scaler=scaler)


def describe(built: BuiltDataset) -> pd.DataFrame:
    """Resumo de cada split: fluxos, ataques e hosts que aparecem nas arestas."""
    rows = {}
    for name in ("train", "val", "test"):
        g = getattr(built.splits, name)
        rows[name] = {
            "fluxos": g.num_edges,
            "ataques": int(g.edge_label.sum()),
            "hosts_com_arestas": int(torch.unique(g.edge_index).numel()),
        }
    return pd.DataFrame(rows).T
