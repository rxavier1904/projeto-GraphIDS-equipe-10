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
"""

from __future__ import annotations

import torch

from graphids.data.types import FlowGraph


class NeighborhoodSampler:
    """Calcula, para cada nó, a média das features das arestas de entrada.

    Args:
        graph: grafo de um split (treino, validação ou teste).
        fanout: máximo de arestas de entrada usadas por nó. ``-1`` usa todas.
    """

    def __init__(self, graph: FlowGraph, fanout: int = -1) -> None:
        self.graph = graph
        self.fanout = fanout
        device = graph.edge_attr.device
        dst = graph.edge_index[1]
        n = graph.num_nodes

        # Arestas ordenadas pelo destino (formato CSC): as arestas de entrada
        # do nó v ficam em in_edges[ptr[v]:ptr[v+1]].
        self.in_edges = torch.argsort(dst, stable=True)
        in_degree = torch.bincount(dst, minlength=n)
        self.ptr = torch.zeros(n + 1, dtype=torch.long, device=device)
        self.ptr[1:] = torch.cumsum(in_degree, 0)
        self.in_degree = in_degree

        # Média sobre a vizinhança completa. Nós sem arestas de entrada ficam
        # com vetor nulo, como no scatter-mean do PyG/DGL.
        total = torch.zeros(n, graph.num_features, device=device)
        total.index_add_(0, dst, graph.edge_attr)
        self.full_mean = total / in_degree.clamp(min=1).unsqueeze(1)

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

    def node_features(
        self, nodes: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Média das arestas de entrada de cada nó de ``nodes``: [len(nodes), F].

        Nós com grau de entrada até o ``fanout`` usam a média pré-calculada;
        os demais são reamostrados a cada chamada.
        """
        feats = self.full_mean[nodes]
        heavy_pos = torch.nonzero(self._is_heavy[nodes]).flatten()
        for pos in heavy_pos.tolist():
            edges = self.sample_incoming(int(nodes[pos]), generator)
            feats[pos] = self.graph.edge_attr[edges].mean(0)
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
