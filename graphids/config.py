"""Configuração de um experimento (contrato C3).

Cada experimento é descrito por um YAML plano, com uma chave por hiperparâmetro.
Os nomes seguem a Tabela 14 do artigo. Valores ausentes no YAML ficam com os
padrões abaixo, e qualquer chave pode ser sobrescrita pela linha de comando
(``--set chave=valor``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from graphids.data.types import DataConfig


@dataclass
class Config:
    """Todos os hiperparâmetros e opções de execução de um experimento."""

    # --- Dados (consumidos pela F2 via DataConfig) ---
    dataset: str = "synthetic"
    data_dir: str = "data"
    fraction: float | None = None  # amostra estratificada (0.2 no CIC-IDS2018)
    drop_timestamps: bool = True  # Seção 4.2: timestamps dos v3 são descartados
    seed: int = 1

    # --- E-GraphSAGE (Seção 3.2) ---
    edim_out: int = 64  # dimensão do embedding de cada fluxo (d_gnn)
    nhops: int = 1  # o artigo usa só 1 hop (Apêndice C.5)
    fanout: int = 32768  # máximo de arestas de entrada amostradas por nó (-1 = todas)
    agg_type: str = "mean"
    gnn_dropout: float = 0.5

    # --- Transformer masked autoencoder (Seção 3.3) ---
    num_layers: int = 1
    num_heads: int = 4
    embed_dim: int = 32  # d_model, após a projeção linear
    ff_dim: int = 256  # dimensão da FFN de cada bloco
    window_size: int = 512  # n <= 512 fluxos por janela
    mask_ratio: float = 0.15  # proporção de ligações de atenção desligadas no treino
    ae_dropout: float = 0.0
    detach_target: bool = False  # o código oficial NÃO corta o gradiente do alvo

    # --- Otimização (Seção 4.3) ---
    learning_rate: float = 1e-4
    gnn_weight_decay: float = 0.6
    ae_weight_decay: float = 0.04
    gnn_batch_size: int = 16384  # arestas por mini-batch do GNN
    ae_batch_size: int = 64  # janelas por mini-batch do Transformer
    num_epochs: int = 100
    patience: int = 20  # early stopping pela PR-AUC de validação
    grad_clip: float = 1.0
    threshold_steps: int = 500  # candidatos de threshold avaliados na validação

    # --- Execução ---
    device: str = "auto"  # "auto", "cpu" ou "cuda"
    output_dir: str = "results/runs"
    extra: dict[str, Any] = field(default_factory=dict)

    def data_config(self) -> DataConfig:
        """Recorta só os campos que o pipeline de dados (F2) precisa."""
        return DataConfig(
            dataset=self.dataset,
            data_dir=self.data_dir,
            fraction=self.fraction,
            seed=self.seed,
            drop_timestamps=self.drop_timestamps,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(value: str, current: Any) -> Any:
    """Converte um valor vindo da linha de comando para o tipo do campo."""
    if value.lower() in {"none", "null"}:
        return None
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "sim"}
    if isinstance(current, int):
        return int(float(value))
    if isinstance(current, float) or current is None:
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None):
    """Lê um YAML de experimento e aplica sobrescritas ``chave=valor``.

    Chaves desconhecidas geram erro, para que um erro de digitação no YAML não
    passe despercebido e rode com o valor padrão.
    """
    cfg = Config()
    known = {f.name for f in dataclasses.fields(Config)}

    if path is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(f"Chaves desconhecidas em {path}: {sorted(unknown)}")
        for key, value in raw.items():
            setattr(cfg, key, value)

    for item in overrides or []:
        key, _, value = item.partition("=")
        if key not in known:
            raise KeyError(f"Chave desconhecida em --set: {key!r}")
        setattr(cfg, key, _coerce(value, getattr(cfg, key)))

    if cfg.nhops != 1:
        raise NotImplementedError("Por enquanto só nhops=1 (configuração do artigo).")
    if cfg.agg_type != "mean":
        raise NotImplementedError("Por enquanto só agg_type='mean' (Tabela 14).")
    return cfg
