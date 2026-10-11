"""Ponto de entrada único para obter os grafos de um dataset (contrato C1)."""

from __future__ import annotations

from graphids.data.synthetic import make_synthetic_splits
from graphids.data.types import DataConfig, GraphSplits


def load_splits(cfg: DataConfig) -> GraphSplits:
    """Devolve os grafos de treino (só benigno), validação e teste.

    ``dataset="synthetic"`` gera um grafo pequeno em memória, usado nos testes e
    no desenvolvimento do modelo. Os datasets NetFlow reais são implementados
    pela F2 (``docs/frentes/F2-dados-e-grafos.md``), que deve substituir o
    ``NotImplementedError`` abaixo mantendo esta assinatura.
    """
    if cfg.dataset == "synthetic":
        return make_synthetic_splits(seed=cfg.seed)
    raise NotImplementedError(
        f"Dataset {cfg.dataset!r} ainda não suportado: o pipeline dos CSVs "
        "NetFlow é a entrega 2.1-2.4 da F2."
    )
