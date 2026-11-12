"""Métricas de avaliação (contrato C4, entrega 4.2 da F4).

Todas as funções recebem arrays NumPy com um elemento por fluxo (aresta) do
conjunto avaliado:

- ``labels``: 0 = benigno, 1 = ataque;
- ``scores``: score de anomalia (erro de reconstrução); maior = mais suspeito;
- ``attacks``: índice da categoria de ataque, que indexa ``attack_names``.

São as mesmas definições usadas no laço de treino (``average_precision_score``
para a PR-AUC e F1 macro com ``zero_division=0``), para que os números das
figuras e das tabelas batam com os de ``metrics.json``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Área sob a curva precisão-revocação (average precision, como no artigo)."""
    return float(average_precision_score(labels, scores))


def f1_macro(labels: np.ndarray, preds: np.ndarray) -> float:
    """F1 médio entre as classes benigna e ataque (sem ponderar pelo tamanho)."""
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def predict(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Classifica como ataque (1) todo fluxo com score acima do threshold.

    Usa ``>`` estrito, igual a ``graphids/training/run.py``.
    """
    return (np.asarray(scores) > threshold).astype(int)


def confusion(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Matriz de confusão 2x2: linhas = verdadeiro, colunas = predito (0, 1)."""
    return confusion_matrix(labels, preds, labels=[0, 1])


def detection_rate_per_attack(
    scores: np.ndarray,
    attacks: np.ndarray,
    threshold: float,
    attack_names: Sequence[str],
) -> dict[str, float]:
    """Fração de fluxos de cada categoria marcada como ataque (score > threshold).

    Para a categoria benigna o valor é a taxa de falso positivo; para as demais,
    é a taxa de detecção (revocação) daquele ataque, como nas Figuras 5 a 8 do
    artigo. Categorias sem nenhum fluxo no conjunto ficam de fora do resultado.
    """
    scores = np.asarray(scores)
    attacks = np.asarray(attacks)
    flagged = scores > threshold
    rates: dict[str, float] = {}
    for idx, name in enumerate(attack_names):
        mask = attacks == idx
        if mask.any():
            rates[str(name)] = float(flagged[mask].mean())
    return rates


def attack_counts(attacks: np.ndarray, attack_names: Sequence[str]) -> dict[str, int]:
    """Número de fluxos de cada categoria (para anotar figuras e tabelas)."""
    counts = np.bincount(np.asarray(attacks), minlength=len(attack_names))
    pairs = zip(attack_names, counts[: len(attack_names)], strict=True)
    return {str(n): int(c) for n, c in pairs if c > 0}
