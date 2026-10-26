"""Ponto de entrada único para obter os grafos de um dataset (contrato C1)."""

from __future__ import annotations

from graphids.data.cache import load_cache, save_cache
from graphids.data.graph import build_graphs
from graphids.data.loading import FlowTable, load_flow_table
from graphids.data.split import sample_and_split
from graphids.data.synthetic import make_synthetic_splits
from graphids.data.types import DataConfig, GraphSplits


def load_splits(
    cfg: DataConfig, use_cache: bool = True, table: FlowTable | None = None
) -> GraphSplits:
    """Devolve os grafos de treino (só benigno), validação e teste.

    ``dataset="synthetic"`` gera um grafo pequeno em memória, usado nos testes e
    no desenvolvimento do modelo. Os demais nomes são datasets NetFlow lidos de
    ``<data_dir>/<dataset>/<dataset>.csv`` pelo pipeline da F2:

    1. leitura e limpeza (``loading.py``, Seção 4.2);
    2. amostra opcional e split 80/10/10 estratificado por ataque, com o treino
       só benigno (``split.py``);
    3. Min-Max ajustado no treino e clip em [−10, 10] (``preprocessing.py``);
    4. grafo dirigido de hosts com mapa de IPs compartilhado (``graph.py``).

    Args:
        cfg: dataset, pasta, ``fraction``, ``seed`` e ``drop_timestamps``.
        use_cache: se verdadeiro, reaproveita ``<data_dir>/cache/...`` quando
            existir e grava o cache depois de processar (``cache.py``).
        table: tabela já lida com ``load_flow_table``, para não ler o CSV de
            novo (usado por ``stats.py``). Se ``None``, o CSV é lido aqui.
    """
    if cfg.dataset == "synthetic":
        return make_synthetic_splits(seed=cfg.seed)

    if use_cache:
        cached = load_cache(cfg)
        if cached is not None:
            return cached

    if table is None:
        table = load_flow_table(cfg.data_dir, cfg.dataset, cfg.drop_timestamps)
    table, indices = sample_and_split(table, seed=cfg.seed, fraction=cfg.fraction)
    built = build_graphs(table, indices)
    if use_cache:
        save_cache(built, cfg)
    return built.splits
