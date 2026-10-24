"""Leitura e limpeza dos CSVs NetFlow (entrega 2.1 da F2).

Segue a Seção 4.2 do artigo e o pré-processamento do código oficial:

1. lê ``<data_dir>/<dataset>/<dataset>.csv``;
2. troca ±inf e NaN por 0 nas features;
3. separa ``Label`` e ``Attack`` das features;
4. tira os IPs das features (eles viram os nós do grafo na entrega 2.4);
5. nos datasets v3, tira também os timestamps, que só adicionam ruído
   (Apêndice C.1 do artigo).

O NF-CSE-CIC-IDS2018 tem cerca de 20 M de linhas. Para caber na memória, cada
coluna é lida já com o tipo final: features em ``float32`` (metade do
``float64`` padrão do pandas) e IPs e categorias de ataque como ``category``,
que guarda cada texto uma única vez.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Nomes das colunas nos datasets NetFlow da UQ (v2 e v3).
LABEL_COL = "Label"  # 0 = benigno, 1 = ataque
ATTACK_COL = "Attack"  # categoria do ataque ("Benign" nos fluxos benignos)
SRC_IP_COL = "IPV4_SRC_ADDR"
DST_IP_COL = "IPV4_DST_ADDR"
IP_COLS = (SRC_IP_COL, DST_IP_COL)
# Só existem nos v3. Seção 4.2: "For v3 datasets, timestamps are discarded".
TIMESTAMP_COLS = ("FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS")
BENIGN = "Benign"


@dataclass
class FlowTable:
    """Fluxos de um dataset já limpos, ainda sem split nem normalização.

    Atributos (E = número de fluxos):
        flows: DataFrame com E linhas. Contém as colunas de IP, as features,
            ``Label`` e ``Attack``. Os índices vão de 0 a E-1.
        feature_names: colunas de ``flows`` que são features do fluxo, na
            ordem em que vão virar as colunas de ``edge_attr``.
    """

    flows: pd.DataFrame
    feature_names: list[str]

    @property
    def num_flows(self) -> int:
        return len(self.flows)

    def features(self) -> np.ndarray:
        """Matriz de features ``[E, F]`` em ``float32``."""
        return self.flows[self.feature_names].to_numpy(dtype=np.float32)


def csv_path(data_dir: str | Path, dataset: str) -> Path:
    """Caminho do CSV de um dataset: ``<data_dir>/<dataset>/<dataset>.csv``.

    Levanta ``FileNotFoundError`` com uma mensagem que diz onde o arquivo era
    esperado, para facilitar a configuração no servidor.
    """
    path = Path(data_dir) / dataset / f"{dataset}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV do dataset {dataset!r} não encontrado em {path}. Os arquivos "
            "devem seguir o padrão <data_dir>/<dataset>/<dataset>.csv (o "
            "NF-CICIDS2018 do site da UQ precisa ser renomeado para "
            "NF-CSE-CIC-IDS2018)."
        )
    return path


def _column_dtypes(columns: list[str]) -> dict[str, str]:
    """Tipo de cada coluna, para o ``read_csv`` não precisar adivinhar."""
    dtypes = {}
    for col in columns:
        if col in IP_COLS or col == ATTACK_COL:
            dtypes[col] = "category"
        elif col == LABEL_COL:
            dtypes[col] = "int8"
        else:
            dtypes[col] = "float32"
    return dtypes


def read_netflow_csv(path: str | Path, drop_timestamps: bool = True) -> FlowTable:
    """Lê um CSV NetFlow e devolve os fluxos limpos.

    Args:
        path: caminho do CSV.
        drop_timestamps: se verdadeiro, as colunas de timestamp dos v3 nem são
            carregadas. Nos v2 elas não existem e a opção não tem efeito.

    Returns:
        ``FlowTable`` com as colunas de IP, as features, ``Label`` e ``Attack``.
    """
    # Lê só o cabeçalho para montar os tipos e escolher as colunas.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in (*IP_COLS, LABEL_COL, ATTACK_COL) if c not in header]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes em {path}: {missing}")

    skip = set(TIMESTAMP_COLS) if drop_timestamps else set()
    usecols = [c for c in header if c not in skip]
    flows = pd.read_csv(path, usecols=usecols, dtype=_column_dtypes(usecols))

    # Features = tudo menos IPs, Label e Attack. Portas, PROTOCOL e L7_PROTO
    # entram como números contínuos, como no código oficial.
    non_features = {*IP_COLS, LABEL_COL, ATTACK_COL}
    feature_names = [c for c in usecols if c not in non_features]

    # ±inf e NaN viram 0 (Seção 4.2: "we replace missing or invalid values
    # with zeros"). Coluna por coluna, para não duplicar a tabela inteira na
    # memória.
    for col in feature_names:
        values = flows[col].to_numpy(dtype=np.float32)
        flows[col] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return FlowTable(flows=flows.reset_index(drop=True), feature_names=feature_names)


def load_flow_table(
    data_dir: str | Path, dataset: str, drop_timestamps: bool = True
) -> FlowTable:
    """Atalho: acha o CSV do dataset e lê com ``read_netflow_csv``."""
    return read_netflow_csv(csv_path(data_dir, dataset), drop_timestamps)
