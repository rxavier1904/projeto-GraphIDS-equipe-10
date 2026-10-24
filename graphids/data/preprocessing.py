"""Normalização das features.

Segue a Seção 4.2 do artigo: "We normalize all NetFlow features to the
interval [0, 1] using Min-Max scaling. The scaler is fitted on the training
set, which contains only benign flows, and then applied to transform both the
validation and test sets. To avoid extreme input values, any scaled value
falling outside the original training range is clipped to the [−10, 10]
range."

Ajustar o scaler só no treino evita vazamento de informação da validação e do
teste. Como o treino só tem fluxos benignos, um ataque com valores muito acima
do normal sai de [0, 1] depois da normalização; o clip em [−10, 10] mantém o
sinal de "fora do normal" sem deixar valores gigantes desestabilizarem o
modelo.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from graphids.data.loading import FlowTable

CLIP_RANGE = (-10.0, 10.0)


def features_at(table: FlowTable, rows: np.ndarray) -> np.ndarray:
    """Matriz de features ``[len(rows), F]`` em ``float32`` só das linhas pedidas.

    Monta a matriz coluna por coluna, sem copiar a tabela inteira antes de
    selecionar as linhas. No NF-CSE-CIC-IDS2018 isso evita alguns GB de pico
    de memória.
    """
    out = np.empty((len(rows), len(table.feature_names)), dtype=np.float32)
    for j, col in enumerate(table.feature_names):
        out[:, j] = table.flows[col].to_numpy(dtype=np.float32)[rows]
    return out


def fit_scaler(train_features: np.ndarray) -> MinMaxScaler:
    """Ajusta um ``MinMaxScaler`` nas features do treino (só benignos).

    Args:
        train_features: ``[E_train, F]``, features do split de treino.

    Returns:
        Scaler com ``data_min_`` e ``data_max_`` do treino. Colunas constantes
        no treino recebem escala 1 (comportamento padrão do scikit-learn).
    """
    return MinMaxScaler(feature_range=(0.0, 1.0)).fit(train_features)


def transform_features(scaler: MinMaxScaler, features: np.ndarray) -> np.ndarray:
    """Normaliza com o scaler do treino e corta em [−10, 10].

    Args:
        scaler: scaler devolvido por ``fit_scaler``.
        features: ``[E, F]``, features de qualquer split.

    Returns:
        ``[E, F]`` em ``float32``. No treino os valores ficam em [0, 1]; na
        validação e no teste, em [−10, 10].
    """
    scaled = scaler.transform(features).astype(np.float32, copy=False)
    # No treino o clip não muda nada (já está em [0, 1]); aplicar nos três
    # splits deixa o código mais simples.
    np.clip(scaled, *CLIP_RANGE, out=scaled)
    return scaled
