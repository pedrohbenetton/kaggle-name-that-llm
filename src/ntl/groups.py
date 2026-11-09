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


def capped_groups(
    texts: Sequence[str], cap: int = 18, threshold: float = 0.05, max_df: float = 1.0
) -> np.ndarray:
    """Agrupamento hierárquico (ligação média) que nunca forma grupo maior que `cap` e para
    quando a melhor similaridade média cai abaixo de `threshold`. Diferente de `infer_groups`,
    não supõe n/18 grupos: grupos parciais (teste desbalanceado) não são fundidos à força.
    Ideia do teto + limiar vista nos notebooks públicos da competição; implementação própria."""
    n = len(texts)
    if n < 2:
        return np.zeros(n, dtype=int)
    try:
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_df=max_df).fit_transform(
            texts
        )
    except ValueError:  # vocabulário vazio
        return np.arange(n)
    A = (X @ X.T).toarray().astype(np.float32)
    np.fill_diagonal(A, -np.inf)
    size = np.ones(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    lab = np.arange(n)

    def row_best(k: int) -> tuple[float, int]:
        ok = alive & (size + size[k] <= cap)
        ok[k] = False
        if not ok.any():
            return -np.inf, -1
        r = np.where(ok, A[k], -np.inf)
        j = int(np.argmax(r))
        return float(r[j]), j

    best = np.full(n, -np.inf, dtype=np.float64)
    arg = np.full(n, -1)
    for k in range(n):
        best[k], arg[k] = row_best(k)

    while True:
        i = int(np.argmax(best))
        if not np.isfinite(best[i]) or best[i] < threshold:
            break
        j = int(arg[i])
        si, sj = size[i], size[j]
        new = (si * A[i] + sj * A[j]) / (si + sj)
        A[i], A[:, i] = new, new
        A[j], A[:, j] = -np.inf, -np.inf
        A[i, i] = -np.inf
        size[i] += sj
        alive[j] = False
        best[j], arg[j] = -np.inf, -1
        lab[lab == j] = i
        # linhas cujo melhor parceiro sumiu ou cresceu demais: recalcula
        for k in np.flatnonzero(alive & ((arg == i) | (arg == j))):
            best[k], arg[k] = row_best(int(k))
        best[i], arg[i] = row_best(i)
        # o grupo novo pode virar o melhor parceiro de outras linhas
        cand = alive & (size + size[i] <= cap) & (A[:, i] > best)
        cand[i] = False
        best[cand], arg[cand] = A[cand, i], i
    return np.unique(lab, return_inverse=True)[1]
