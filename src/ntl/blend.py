"""Classificador sobre embeddings e mistura de log-probabilidades de modelos diferentes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def emb_lr(C: float = 0.05) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=3000))])


def blend_logp(Ls: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    return sum(w * L for w, L in zip(weights, Ls, strict=True))


def choose_weight(La, Lb, y, classes, grid=None) -> tuple[float, float]:
    """Peso w de La (1-w para Lb) com maior balanced accuracy; empate fica com o maior w."""
    grid = np.round(np.linspace(0, 1, 11), 2) if grid is None else grid
    best = (-1.0, 0.0)
    for w in grid:
        p = np.asarray(classes)[blend_logp([La, Lb], [w, 1 - w]).argmax(1)]
        s = float(balanced_accuracy_score(y, p))
        if s >= best[0]:
            best = (s, float(w))
    return best[1], best[0]


def choose_weights(Ls, y, classes, step: float = 0.1) -> tuple[tuple[float, ...], float]:
    """Pesos (somando 1, passo `step`) da mistura de log-probabilidades com maior balanced
    accuracy. Um modelo que piora a mistura recebe peso 0, então ele não atrapalha."""
    k = round(1 / step)
    classes = np.asarray(classes)
    best: tuple[float, tuple[float, ...]] = (-1.0, ())

    def grid(n, left):
        if n == 1:
            yield (left,)
            return
        for i in range(left + 1):
            for rest in grid(n - 1, left - i):
                yield (i, *rest)

    for combo in grid(len(Ls), k):
        w = tuple(c / k for c in combo)
        s = float(balanced_accuracy_score(y, classes[blend_logp(Ls, w).argmax(1)]))
        if s > best[0]:
            best = (s, w)
    return best[1], best[0]
