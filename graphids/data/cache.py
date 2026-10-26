"""Cache dos grafos já construídos.

Ler e processar o NF-CSE-CIC-IDS2018 leva minutos. O cache grava o resultado
em ``<data_dir>/cache/<dataset>[_<fraction>]_seed<s>/`` para que as próximas
execuções com a mesma configuração carreguem os grafos direto:

    graphs.pt    tensores dos três grafos (torch.save)
    meta.json    features, categorias de ataque, número de nós e versão
    hosts.txt    IP de cada índice de nó, um por linha
    scaler.pkl   Min-Max ajustado no treino (pickle do scikit-learn)

Se a lógica do pipeline mudar de um jeito que altere os grafos, aumente
``CACHE_VERSION``: caches antigos deixam de ser usados automaticamente.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

from graphids.data.graph import BuiltDataset
from graphids.data.types import DataConfig, FlowGraph, GraphSplits

CACHE_VERSION = 2
_TENSORS = ("edge_index", "edge_attr", "edge_label", "edge_attack")
_SPLITS = ("train", "val", "test")


def cache_dir(cfg: DataConfig) -> Path:
    """Pasta do cache de uma configuração de dados.

    Exemplos: ``data/cache/NF-UNSW-NB15-v3_seed1`` e
    ``data/cache/NF-CSE-CIC-IDS2018-v3_0.2_seed1``. Quando os timestamps são
    mantidos, a pasta ganha o sufixo ``_ts`` para não reaproveitar um cache
    feito sem eles.
    """
    name = cfg.dataset
    if cfg.fraction is not None and cfg.fraction < 1:
        name += f"_{cfg.fraction:g}"
    name += f"_seed{cfg.seed}"
    if not cfg.drop_timestamps:
        name += "_ts"
    return Path(cfg.data_dir) / "cache" / name


def save_cache(built: BuiltDataset, cfg: DataConfig) -> Path:
    """Grava os grafos, os metadados, os hosts e o scaler. Devolve a pasta."""
    folder = cache_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    splits = built.splits

    tensors = {
        split: {key: getattr(getattr(splits, split), key) for key in _TENSORS}
        for split in _SPLITS
    }
    torch.save(tensors, folder / "graphs.pt")

    meta = {
        "cache_version": CACHE_VERSION,
        "dataset": cfg.dataset,
        "fraction": cfg.fraction,
        "seed": cfg.seed,
        "drop_timestamps": cfg.drop_timestamps,
        "num_nodes": splits.train.num_nodes,
        "feature_names": splits.feature_names,
        "attack_names": splits.train.attack_names,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (folder / "hosts.txt").write_text("\n".join(built.hosts) + "\n")
    with open(folder / "scaler.pkl", "wb") as f:
        pickle.dump(built.scaler, f)
    return folder


def load_cache(cfg: DataConfig) -> GraphSplits | None:
    """Carrega os grafos do cache, ou ``None`` se não houver cache válido."""
    folder = cache_dir(cfg)
    meta_path = folder / "meta.json"
    graphs_path = folder / "graphs.pt"
    if not (meta_path.is_file() and graphs_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("cache_version") != CACHE_VERSION:
        return None

    # weights_only=True: só aceita tensores e tipos básicos (mais seguro).
    tensors = torch.load(graphs_path, weights_only=True)
    graphs = {
        split: FlowGraph(
            **tensors[split],
            num_nodes=meta["num_nodes"],
            attack_names=list(meta["attack_names"]),
        )
        for split in _SPLITS
    }
    return GraphSplits(**graphs, feature_names=list(meta["feature_names"]))
