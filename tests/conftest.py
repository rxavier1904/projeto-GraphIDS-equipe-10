"""Fixtures compartilhadas: grafos sintéticos pequenos e uma config rápida."""

import pytest
import torch

from graphids.config import load_config
from graphids.data.synthetic import make_synthetic_splits
from graphids.data.types import FlowGraph


@pytest.fixture
def splits():
    return make_synthetic_splits(seed=0)


@pytest.fixture
def tiny_graph():
    """4 hosts, 6 fluxos. O nó 2 recebe 3 fluxos; os nós 0 e 3 não recebem nada."""
    edge_index = torch.tensor([[0, 1, 3, 0, 1, 2], [2, 2, 2, 1, 0, 1]])
    edge_attr = torch.arange(12, dtype=torch.float).view(6, 2)
    zeros = torch.zeros(6, dtype=torch.long)
    return FlowGraph(edge_index, edge_attr, zeros, zeros, num_nodes=4)


@pytest.fixture
def fast_cfg(tmp_path):
    cfg = load_config("configs/synthetic.yaml")
    cfg.output_dir = str(tmp_path / "runs")
    cfg.device = "cpu"
    torch.manual_seed(0)
    return cfg
