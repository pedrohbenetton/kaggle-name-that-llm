"""Infere os grupos de prompt só pelo conteúdo: textos que respondem ao mesmo prompt
compartilham vocabulário de tema. Usa TF-IDF de palavras (sem stopwords) e agrupamento
hierárquico por cosseno."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


def infer_groups(
    texts: Sequence[str], n_groups: int | None = None, per_group: int = 18
) -> np.ndarray:
    if len(texts) < 2:
        return np.zeros(len(texts), dtype=int)
    if n_groups is None:
        n_groups = max(1, round(len(texts) / per_group))
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1).fit_transform(texts)
    # distância de cosseno pré-computada (n x n): evita densificar a matriz n x vocabulário
    D = np.clip(1.0 - (X @ X.T).toarray(), 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    n_groups = min(n_groups, len(texts))
    clu = AgglomerativeClustering(n_clusters=n_groups, metric="precomputed", linkage="average")
    return clu.fit_predict(D)
