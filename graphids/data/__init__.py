"""Pipeline de dados: dos CSVs NetFlow aos grafos de treino, validação e teste.

A leitura dos datasets reais é da F2 (Danilo). Aqui ficam o contrato C1
(``types.py``), o gerador sintético usado nos testes e o ponto de entrada
``load_splits``.
"""

from graphids.data.loader import load_splits
from graphids.data.types import DataConfig, FlowGraph, GraphSplits

__all__ = ["DataConfig", "FlowGraph", "GraphSplits", "load_splits"]
