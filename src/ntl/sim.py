"""Simula o teste na validação. A página diz que no teste o número de textos por modelo não é
igual; no treino cada prompt tem exatamente um texto de cada modelo. Aqui cada modelo recebe
uma taxa de retenção sorteada, o que deixa modelos com frequências diferentes e grupos parciais."""

from __future__ import annotations

import numpy as np
import pandas as pd


def unbalanced_subsample(
    df: pd.DataFrame, idx: np.ndarray, seed: int = 0, keep: tuple[float, float] = (0.3, 1.0)
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    models = sorted(df["model"].unique())
    rate = dict(zip(models, rng.uniform(*keep, size=len(models)), strict=True))
    idx = np.asarray(idx)
    p = df["model"].to_numpy()[idx]
    mask = rng.random(len(idx)) < np.array([rate[m] for m in p])
    return idx[mask]
