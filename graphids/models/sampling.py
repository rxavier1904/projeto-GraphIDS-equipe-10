"""Amostragem de vizinhança de 1 hop para o E-GraphSAGE (Seção 3.2 do artigo).

No E-GraphSAGE de 1 camada, a representação de um host v depende só da média
das features dos fluxos que CHEGAM em v. Como essa média é feita sobre as
features de entrada (antes de qualquer peso treinável), ela não precisa de
gradiente. Por isso dá para:

1. pré-calcular a média sobre TODAS as arestas de entrada de cada nó;
2. em cada mini-batch, refazer a média só para os nós cujo grau de entrada
   passa do ``fanout``, sorteando ``fanout`` arestas sem reposição.

O resultado é equivalente ao ``LinkNeighborLoader`` usado no código oficial,
sem depender do PyTorch Geometric.

Além da média do artigo (``agg_type="mean"``), há variantes experimentais da
agregação, usadas só na ablação de agregação (fora da Tabela 14):

- ``wmean``: média ponderada por ``w = log1p(bytes / mediana_bytes_treino)``,
  com ``bytes = IN_BYTES + OUT_BYTES`` (Min-Max é linear, então a razão pela
  mediana do treino equivale à razão nos bytes brutos, a menos do mínimo);
- ``max``: máximo de cada feature sobre as arestas de entrada;
- ``sum``: soma das arestas de entrada, comprimida com ``sign(x)·log1p(|x|)``
  para não explodir em hosts com milhões de fluxos. Com fanout, a soma é
  estimada como média amostrada × grau de entrada;
- ``mean_max``: concatenação ``[média ‖ máximo]`` (dobra a dimensão do host).
"""

from __future__ import annotations

import numpy as np
import torch

from graphids.data.types import FlowGraph, GraphSplits

AGG_TYPES = ("mean", "wmean", "max", "sum", "mean_max")
BYTE_FEATURES = ("IN_BYTES", "OUT_BYTES")


def agg_dim_multiplier(agg_type: str) -> int:
    """Quantas vezes F tem o vetor agregado de cada host."""
    if agg_type not in AGG_TYPES:
        raise ValueError(f"agg_type desconhecido: {agg_type!r} (opções: {AGG_TYPES})")
    return 2 if agg_type == "mean_max" else 1


def _bytes(graph: FlowGraph, feature_names: list[str]) -> torch.Tensor:
    missing = [c for c in BYTE_FEATURES if c not in feature_names]
    if missing:
        raise KeyError(f"wmean precisa das features {missing}")
    cols = [feature_names.index(c) for c in BYTE_FEATURES]
    return graph.edge_attr[:, cols].sum(1).clamp(min=0)


def byte_reference(splits: GraphSplits) -> float:
    """Mediana (positiva) dos bytes normalizados do TREINO, escala dos pesos."""
    b = _bytes(splits.train, splits.feature_names)
    pos = b[b > 0]
    return float(pos.median()) if pos.numel() else 1.0


def edge_weights(graph: FlowGraph, feature_names: list[str], ref: float):
    """Peso de cada fluxo na média ponderada: ``1e-3 + log1p(bytes / ref)``."""
    return 1e-3 + torch.log1p(_bytes(graph, feature_names) / ref)


def make_sampler(graph: FlowGraph, cfg, splits: GraphSplits) -> NeighborhoodSampler:
    """Cria o sampler do ``graph`` com a agregação de ``cfg.agg_type``.

    Os pesos do ``wmean`` usam a escala do treino em todos os splits.
    """
    weight = None
    if cfg.agg_type == "wmean":
        ref = byte_reference(splits)
        weight = edge_weights(graph, splits.feature_names, ref)
    mask = None
    if cfg.prune_bytes_pct is not None:
        mask = prune_mask(graph, splits, cfg.prune_bytes_pct)
    return NeighborhoodSampler(graph, cfg.fanout, cfg.agg_type, weight, mask)


def prune_mask(graph: FlowGraph, splits: GraphSplits, pct: float) -> torch.Tensor:
    """Arestas mantidas na agregação: bytes >= percentil ``pct`` do TREINO.

    O corte vem só do treino (benigno), sem rótulos de ataque.
    """
    train_bytes = _bytes(splits.train, splits.feature_names).cpu().numpy()
    thr = float(np.percentile(train_bytes, pct))
    return _bytes(graph, splits.feature_names) >= thr


def _slog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


class NeighborhoodSampler:
    """Calcula, para cada nó, a agregação das features das arestas de entrada.

    Args:
        graph: grafo de um split (treino, validação ou teste).
        fanout: máximo de arestas de entrada usadas por nó. ``-1`` usa todas.
        agg_type: ``mean`` (artigo) ou uma variante de ``AGG_TYPES``.
        edge_weight: [E] pesos dos fluxos, obrigatório em ``wmean``.
        edge_mask: [E] bool. Se dado, só as arestas ``True`` entram na
            agregação dos hosts (poda da vizinhança). Todos os fluxos continuam
            recebendo embedding e score.
    """

    def __init__(
        self,
        graph: FlowGraph,
        fanout: int = -1,
        agg_type: str = "mean",
        edge_weight: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> None:
        agg_dim_multiplier(agg_type)  # valida o nome
        if agg_type == "wmean" and edge_weight is None:
            raise ValueError(
                "agg_type='wmean' precisa de edge_weight (use make_sampler)"
            )
        self.graph = graph
        self.fanout = fanout
        self.agg_type = agg_type
        self.edge_weight = edge_weight
        device = graph.edge_attr.device
        dst = graph.edge_index[1]
        n = graph.num_nodes
        attr = graph.edge_attr

        # Poda: as estatísticas dos hosts usam só as arestas mantidas. Os ids
        # guardados em in_edges continuam sendo ids globais do grafo.
        if edge_mask is not None:
            kept = torch.nonzero(edge_mask.to(device)).flatten()
            dst, attr = dst[kept], attr[kept]
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)
                w_kept = edge_weight[kept]
        else:
            kept = None
            w_kept = edge_weight
        self.edge_mask = edge_mask

        # Arestas ordenadas pelo destino (formato CSC): as arestas de entrada
        # do nó v ficam em in_edges[ptr[v]:ptr[v+1]].
        order = torch.argsort(dst, stable=True)
        self.in_edges = order if kept is None else kept[order]
        in_degree = torch.bincount(dst, minlength=n)
        self.ptr = torch.zeros(n + 1, dtype=torch.long, device=device)
        self.ptr[1:] = torch.cumsum(in_degree, 0)
        self.in_degree = in_degree
        deg = in_degree.clamp(min=1).unsqueeze(1).to(attr.dtype)

        # Média sobre a vizinhança completa. Nós sem arestas de entrada ficam
        # com vetor nulo, como no scatter-mean do PyG/DGL.
        total = torch.zeros(n, attr.shape[1], device=device)
        total.index_add_(0, dst, attr)
        self.full_mean = total / deg

        # Agregação completa da variante escolhida (= full_mean no padrão).
        if agg_type == "mean":
            self.full_agg = self.full_mean
        elif agg_type == "wmean":
            w = w_kept.to(device=device, dtype=attr.dtype)
            num = torch.zeros_like(total).index_add_(0, dst, attr * w.unsqueeze(1))
            den = torch.zeros(n, device=device, dtype=attr.dtype).index_add_(0, dst, w)
            self.full_agg = num / den.clamp(min=1e-12).unsqueeze(1)
        elif agg_type == "sum":
            self.full_agg = _slog(total)
        else:
            full_max = torch.zeros_like(total)
            full_max.scatter_reduce_(
                0, dst.unsqueeze(1).expand_as(attr), attr, "amax", include_self=False
            )
            self.full_agg = (
                full_max
                if agg_type == "max"
                else torch.cat([self.full_mean, full_max], dim=1)
            )

        if fanout > 0:
            self.heavy = torch.nonzero(in_degree > fanout).flatten()
        else:
            self.heavy = torch.empty(0, dtype=torch.long, device=device)
        self._is_heavy = torch.zeros(n, dtype=torch.bool, device=device)
        self._is_heavy[self.heavy] = True

    def sample_incoming(self, node: int, generator: torch.Generator | None = None):
        """Sorteia ``fanout`` arestas de entrada de ``node``, sem reposição."""
        start, end = int(self.ptr[node]), int(self.ptr[node + 1])
        perm = torch.randperm(end - start, generator=generator)[: self.fanout]
        return self.in_edges[start + perm.to(self.in_edges.device)]

    def aggregate(self, edges: torch.Tensor, node: int) -> torch.Tensor:
        """Agrega as arestas ``edges`` (amostra das arestas de entrada de ``node``)."""
        x = self.graph.edge_attr[edges]
        if self.agg_type == "mean":
            return x.mean(0)
        if self.agg_type == "wmean":
            w = self.edge_weight[edges].to(x.dtype)
            return (x * w.unsqueeze(1)).sum(0) / w.sum().clamp(min=1e-12)
        if self.agg_type == "sum":
            # estimativa da soma completa: média amostrada × grau de entrada
            return _slog(x.mean(0) * self.in_degree[node].to(x.dtype))
        if self.agg_type == "max":
            return x.amax(0)
        return torch.cat([x.mean(0), x.amax(0)])

    def node_features(
        self, nodes: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Agregação das arestas de entrada de cada nó de ``nodes``.

        Devolve ``[len(nodes), F]`` (``[len(nodes), 2F]`` em ``mean_max``).
        Nós com grau de entrada até o ``fanout`` usam a agregação pré-calculada;
        os demais são reamostrados a cada chamada.
        """
        feats = self.full_agg[nodes]
        heavy_pos = torch.nonzero(self._is_heavy[nodes]).flatten()
        for pos in heavy_pos.tolist():
            node = int(nodes[pos])
            edges = self.sample_incoming(node, generator)
            feats[pos] = self.aggregate(edges, node)
        return feats

    def batch(self, edge_ids: torch.Tensor, generator: torch.Generator | None = None):
        """Prepara um mini-batch de fluxos para o E-GraphSAGE.

        Returns:
            node_feats: [N_b, F], média de entrada dos hosts envolvidos no batch.
            src_pos, dst_pos: [B], posição da origem e do destino de cada fluxo
                dentro de ``node_feats``.
        """
        src = self.graph.edge_index[0, edge_ids]
        dst = self.graph.edge_index[1, edge_ids]
        nodes, inverse = torch.unique(torch.cat([src, dst]), return_inverse=True)
        node_feats = self.node_features(nodes, generator)
        b = edge_ids.shape[0]
        return node_feats, inverse[:b], inverse[b:]
