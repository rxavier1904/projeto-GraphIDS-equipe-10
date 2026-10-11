"""Escolha do threshold de decisão (Seção 4.5 do artigo).

O modelo dá um score contínuo para cada fluxo. Para virar classificação binária,
testamos ``steps`` thresholds igualmente espaçados entre o menor e o maior score
da VALIDAÇÃO e ficamos com o que maximiza o F1 macro. O mesmo threshold é depois
aplicado ao teste, sem olhar para ele.

O F1 de todos os candidatos é calculado de uma vez, com contagens acumuladas
(``searchsorted``), em vez de chamar o scikit-learn 500 vezes.
"""

from __future__ import annotations

import numpy as np


def macro_f1_curve(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """F1 macro de ``pred = scores > t`` para cada ``t`` em ``thresholds``.

    Equivale a ``sklearn.metrics.f1_score(labels, pred, average="macro",
    zero_division=0)``, mas vetorizado.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    pos = np.sort(scores[labels])
    neg = np.sort(scores[~labels])
    # quantos positivos/negativos têm score > t (previstos como ataque)
    tp = len(pos) - np.searchsorted(pos, thresholds, side="right")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="right")
    fn = len(pos) - tp
    tn = len(neg) - fp

    def f1(t, f_p, f_n):
        denom = 2 * t + f_p + f_n
        return np.divide(2 * t, denom, out=np.zeros(len(t)), where=denom > 0)

    return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2


def best_f1_threshold(
    scores: np.ndarray, labels: np.ndarray, steps: int = 500
) -> tuple[float, float]:
    """Threshold com maior F1 macro na validação.

    Returns:
        (threshold, f1_macro). Em caso de empate fica o menor threshold, como no
        código oficial.
    """
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.linspace(scores.min(), scores.max(), steps)
    curve = macro_f1_curve(scores, labels, thresholds)
    best = int(np.argmax(curve))
    return float(thresholds[best]), float(curve[best])


def mad_threshold(scores: np.ndarray, multiplier: float = 10.0) -> float:
    """Threshold sem rótulos: mediana + ``multiplier`` · MAD (desvio absoluto).

    Não é usado nos resultados do artigo; fica disponível para comparação.
    """
    scores = np.asarray(scores, dtype=np.float64)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median)) * 1.4826  # fator da normal
    return float(median + multiplier * mad)
