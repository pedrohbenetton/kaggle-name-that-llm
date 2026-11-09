"""Modelos. Baseline: TF-IDF de palavras + caracteres e regressão logística."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def tfidf_lr(max_word: int = 25000, max_char: int = 35000, C: float = 2.0) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=max_word, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=max_char,
                                 sublinear_tf=True)),
    ])
    return Pipeline([("feats", feats), ("clf", LogisticRegression(C=C, max_iter=2000))])
