"""Métricas de avaliação do GraphIDS (contrato C4).

Este módulo implementa as métricas de detecção de anomalia exigidas na avaliação
do modelo:
- PR-AUC: Área sob a curva Precision-Recall (métrica principal de validação).
- F1-Macro: Média não-ponderada do F1 entre as classes benigna e ataque.
- Taxa de detecção por ataque: Proporção de amostras detectadas (score > limiar)
  para cada categoria de ataque individual.
- Matriz de confusão / métricas derivadas (TPR, FPR, precisão, revocação).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score


def pr_auc(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> float:
    """Calcula a área sob a curva Precisão-Revocação (PR-AUC).

    No GraphIDS, esta é a métrica principal usada para selecionar o melhor modelo
    durante a validação (early stopping) e para a Tabela 3 do artigo.

    Args:
        labels: Rótulos binários verdadeiros [N], onde 0 = benigno e 1 = ataque.
        scores: Scores de anomalia [N] (erro de reconstrução do autoencoder).

    Returns:
        Valor escalar da PR-AUC em [0.0, 1.0].
    """
    y_true = np.asarray(labels)
    y_score = np.asarray(scores)
    if len(y_true) == 0:
        return 0.0
    # Caso trivial: apenas uma classe presente (ex.: só benignos)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def f1_macro(
    labels: Sequence[int] | np.ndarray, preds: Sequence[int] | np.ndarray
) -> float:
    """Calcula o F1-score macro (média aritmética entre F1 benigno e F1 ataque).

    Args:
        labels: Rótulos binários verdadeiros [N] (0 = benigno, 1 = ataque).
        preds: Predições binárias [N] (pred = score > threshold).

    Returns:
        Valor escalar do F1 macro em [0.0, 1.0].
    """
    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))


def detection_rate_per_attack(
    scores: Sequence[float] | np.ndarray,
    attacks: Sequence[int] | np.ndarray,
    threshold: float,
    attack_names: Sequence[str],
) -> dict[str, float]:
    """Calcula a taxa de detecção (revocação / TPR) para cada categoria de ataque.

    Para cada classe de ataque presente em ``attack_names`` (excluindo a categoria
    "Benign", que corresponde a falso positivo se detectada), calcula a fração:
    ``DR_k = sum(score_i > threshold) / N_k`` para todo i com attack_i == k.

    Args:
        scores: Erros de reconstrução [N] de cada fluxo.
        attacks: Índices inteiros [N] correspondentes à categoria em ``attack_names``.
        threshold: Limiar de corte ótimo selecionado na validação.
        attack_names: Lista com o nome legível de cada categoria (índice 0 é "Benign").

    Returns:
        Dicionário mapeando o nome de cada ataque à sua respectiva taxa de detecção
        em [0.0, 1.0].
    """
    y_score = np.asarray(scores)
    y_atk = np.asarray(attacks)

    rates: dict[str, float] = {}
    for idx, name in enumerate(attack_names):
        # A taxa de detecção refere-se aos ataques maliciosos (Figuras 5 a 8 do artigo)
        if name.lower() == "benign":
            continue

        mask = y_atk == idx
        total = int(np.sum(mask))
        if total == 0:
            rates[name] = 0.0
        else:
            detected = int(np.sum(y_score[mask] > threshold))
            rates[name] = float(detected / total)

    return rates


def binary_confusion_metrics(
    labels: Sequence[int] | np.ndarray, preds: Sequence[int] | np.ndarray
) -> dict[str, float]:
    """Calcula a matriz de confusão e métricas de diagnóstico binário.

    Args:
        labels: Rótulos reais [N] (0 = benigno, 1 = ataque).
        preds: Predições binárias [N] (0 = benigno, 1 = ataque).

    Returns:
        Dicionário contendo tn, fp, fn, tp, tpr, fpr, precision e recall.
    """
    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)

    if len(y_true) == 0:
        return {
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tp": 0.0,
            "tpr": 0.0,
            "fpr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = tpr

    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
    }
