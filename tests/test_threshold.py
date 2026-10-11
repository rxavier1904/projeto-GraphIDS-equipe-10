import numpy as np
from sklearn.metrics import f1_score

from graphids.training.threshold import best_f1_threshold, macro_f1_curve


def test_curva_igual_ao_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.random(300) < 0.2
    scores = rng.normal(size=300) + 1.5 * labels
    thresholds = np.linspace(scores.min(), scores.max(), 50)
    curve = macro_f1_curve(scores, labels, thresholds)
    expected = [
        f1_score(labels, scores > t, average="macro", zero_division=0)
        for t in thresholds
    ]
    np.testing.assert_allclose(curve, expected, atol=1e-12)


def test_threshold_separa_classes_perfeitas():
    scores = np.array([0.1, 0.2, 0.3, 5.0, 6.0])
    labels = np.array([0, 0, 0, 1, 1])
    t, f1 = best_f1_threshold(scores, labels)
    assert 0.3 <= t < 5.0 and f1 == 1.0
