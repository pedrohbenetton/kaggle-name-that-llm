"""Validação cruzada. O teste usa prompts que não estão no treino, então a validação
honesta agrupa por prompt_id (GroupKFold); a aleatória existe só para medir o vazamento."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

Folds = list[tuple[np.ndarray, np.ndarray]]


def grouped_folds(df: pd.DataFrame, n_splits: int = 5) -> Folds:
    return list(GroupKFold(n_splits=n_splits).split(df, df["model"], groups=df["prompt_id"]))


def random_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> Folds:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df, df["model"]))


def evaluate(factory: Callable[[], object], df: pd.DataFrame, folds: Folds) -> dict:
    """Treina um modelo novo por fold e mede balanced accuracy (a métrica oficial)."""
    X, y = df["text"].to_numpy(), df["model"].to_numpy()
    oof = np.empty(len(df), dtype=object)
    scores = None
    per_fold = []
    for tr, va in folds:
        m = factory()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X[va])
            if scores is None:
                scores = np.zeros((len(df), p.shape[1]))
                classes = list(m.classes_)
            scores[va] = p
        per_fold.append(balanced_accuracy_score(y[va], oof[va]))
    labels = sorted(set(y))
    recalls = recall_score(y, oof, labels=labels, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, oof)),
        "std": float(np.std(per_fold)),
        "per_fold": [float(x) for x in per_fold],
        "per_class_recall": dict(zip(labels, map(float, recalls), strict=True)),
        "oof": oof,
        "oof_proba": scores,
        "classes": classes if scores is not None else None,
    }
