"""Estruturas de dados trocadas entre o pipeline de dados (F2) e o modelo (F1).

Este arquivo É o contrato C1 (ver docs/CONTRATOS.md). Usamos PyTorch puro, sem
PyTorch Geometric, para simplificar a instalação no servidor do CIn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DataConfig:
    """Parâmetros que o pipeline de dados precisa para montar os grafos."""

    dataset: str
    data_dir: str = "data"
    fraction: float | None = None
    seed: int = 1
    drop_timestamps: bool = True


@dataclass
class FlowGraph:
    """Grafo dirigido de hosts (nós) ligados por fluxos NetFlow (arestas).

    Atributos (E = número de fluxos, F = número de features):
        edge_index:  [2, E] long. Linha 0 = host de origem, linha 1 = destino.
        edge_attr:   [E, F] float. Features do fluxo já normalizadas.
        edge_label:  [E] long. 0 = benigno, 1 = ataque.
        edge_attack: [E] long. Índice da categoria em ``attack_names``.
        attack_names: nome de cada categoria (posição 0 deve ser "Benign").
        num_nodes:   número de hosts (mapa de IPs compartilhado entre splits).
    """

    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_label: torch.Tensor
    edge_attack: torch.Tensor
    num_nodes: int
    attack_names: list[str] = field(default_factory=lambda: ["Benign"])

    def __post_init__(self) -> None:
        e = self.edge_attr.shape[0]
        assert self.edge_index.shape == (2, e), "edge_index deve ser [2, E]"
        assert self.edge_label.shape == (e,), "edge_label deve ser [E]"
        assert self.edge_attack.shape == (e,), "edge_attack deve ser [E]"

    @property
    def num_edges(self) -> int:
        return self.edge_attr.shape[0]

    @property
    def num_features(self) -> int:
        return self.edge_attr.shape[1]

    def to(self, device: torch.device | str) -> FlowGraph:
        """Copia os tensores para ``device`` (CPU ou GPU)."""
        return FlowGraph(
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            edge_label=self.edge_label.to(device),
            edge_attack=self.edge_attack.to(device),
            num_nodes=self.num_nodes,
            attack_names=list(self.attack_names),
        )


@dataclass
class GraphSplits:
    """Os três grafos de um dataset. As arestas não cruzam partições (Seção 4.2)."""

    train: FlowGraph  # só fluxos benignos
    val: FlowGraph  # benignos + ataques, rotulado
    test: FlowGraph  # benignos + ataques, rotulado
    feature_names: list[str]

    @property
    def num_features(self) -> int:
        return self.train.num_features
