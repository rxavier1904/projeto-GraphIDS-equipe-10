"""Camada E-GraphSAGE que gera um embedding por fluxo (Seção 3.2 do artigo).

Para um fluxo u -> v:

    a_w  = média das features das arestas que chegam em w   (agregação, sem pesos)
    h_w  = ReLU(W_neigh · a_w)                               (embedding do host)
    h_uv = Dropout(W_edge · [h_u || h_v])                    (embedding do fluxo)

Assim cada fluxo carrega o contexto local (1 hop) dos dois hosts que ele liga.
A feature do próprio nó não entra (os nós não têm atributos nos datasets NetFlow),
e o fluxo só influencia o próprio embedding pela agregação em v. As duas
escolhas seguem o código oficial (ver docs/REFERENCIA-CODIGO-OFICIAL.md).
"""

from __future__ import annotations

import torch
from torch import nn


class EGraphSAGE(nn.Module):
    """E-GraphSAGE de 1 camada com agregação por média.

    Args:
        in_features: número de features de cada fluxo (F).
        out_features: dimensão do embedding do fluxo (``edim_out``).
        dropout: dropout aplicado no embedding final (até 0.75 na Tabela 14).
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc_neigh = nn.Linear(in_features, in_features)
        self.fc_edge = nn.Linear(2 * in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.zeros_(self.fc_neigh.bias)
        nn.init.zeros_(self.fc_edge.bias)

    def forward(
        self, node_feats: torch.Tensor, src_pos: torch.Tensor, dst_pos: torch.Tensor
    ) -> torch.Tensor:
        """Calcula os embeddings dos fluxos de um mini-batch.

        Args:
            node_feats: [N_b, F], média das arestas de entrada de cada host
                (vem de ``NeighborhoodSampler.batch``).
            src_pos, dst_pos: [B], índices da origem e do destino em ``node_feats``.

        Returns:
            [B, out_features], um embedding por fluxo.
        """
        h_nodes = torch.relu(self.fc_neigh(node_feats))
        h_edges = self.fc_edge(torch.cat([h_nodes[src_pos], h_nodes[dst_pos]], dim=1))
        return self.dropout(h_edges)
