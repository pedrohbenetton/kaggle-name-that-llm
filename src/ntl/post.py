"""Pós-processamento por grupo de prompt. Cada prompt costuma ter um texto por modelo, então
(1) centralizar os scores dentro do grupo tira o viés de tema e (2) a atribuição (Hungarian)
impõe no máximo um texto por modelo no grupo."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _groups(groups: Sequence) -> dict:
    idx: dict = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    return idx


def center_by_group(scores: np.ndarray, groups: Sequence) -> np.ndarray:
    out = scores.astype(float).copy()
    for rows in _groups(groups).values():
        out[rows] -= out[rows].mean(axis=0)
    return out


def assign_by_group(scores: np.ndarray, groups: Sequence) -> np.ndarray:
    """Devolve índices de classe. Grupos maiores que o nº de classes ficam com o argmax."""
    pred = scores.argmax(axis=1)
    for rows in _groups(groups).values():
        if len(rows) > scores.shape[1]:
            continue
        r, c = linear_sum_assignment(-scores[rows])
        pred[np.asarray(rows)[r]] = c
    return pred
