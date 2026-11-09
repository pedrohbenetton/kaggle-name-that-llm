"""Modelos. Baseline: TF-IDF de palavras + caracteres e regressão logística.
`four_view_lr` acrescenta duas visões de estilo que ignoram o tema: a "forma" das palavras
(maiúscula/minúscula) e as palavras funcionais. As quatro visões e seus pesos vêm do SVC4 dos
notebooks públicos da competição; a implementação aqui é própria."""

from __future__ import annotations

import re

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


FUNCTION_WORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "while", "although", "because", "as", "so", "than", "that", "which", "who", "whom", "whose", "this", "these", "those", "it", "its", "they", "them", "their", "theirs", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you", "your", "yours", "i", "me", "my", "mine", "in", "on", "at", "to", "from", "with", "without", "for", "of", "by", "over", "under", "between", "through", "into", "about", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must", "not", "no", "nor", "all", "any", "some", "each", "every", "both", "either", "neither", "other", "another", "such", "only", "also", "just", "very", "too", "more", "most", "less", "least", "here", "there", "when", "where", "why", "how", "what", "whether", "then", "now", "still", "yet", "even", "much", "many", "few", "little"]
)
_WORD = re.compile(r"[^\W\d_]+")
_TOKEN = re.compile(r"\w+|[^\w\s]")


def _shape(w: str) -> str:
    head = "X" if w[0].isupper() else "x"
    if len(w) == 1:
        return head
    return head + ("X" if w[1:].isupper() else "x")


def shape_view(text: str) -> str:
    """Troca cada palavra pelo seu formato (Xx, x, XX); mantém pontuação, números e espaços."""
    return _WORD.sub(lambda m: _shape(m[0]), text)


def function_view(text: str) -> str:
    """Mantém palavras funcionais e pontuação; o resto vira CONTENT."""
    return " ".join(
        t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else "CONTENT"
        for t in _TOKEN.findall(text)
    )


VIEWS = (  # nome, analisador, n-gramas, máx. de features, peso
    ("char", "char", (3, 5), 250000, 1.0),
    ("word", "word", (1, 3), 180000, 0.7),
    ("shape", "char", (2, 6), 60000, 0.5),
    ("function", "word", (1, 3), 50000, 0.5),
)
_PRE = {"shape": shape_view, "function": function_view}


def four_view_features(max_features: int | None = None, min_df: int = 2) -> FeatureUnion:
    parts, weights = [], {}
    for name, analyzer, ngrams, cap, w in VIEWS:
        opts = {"analyzer": analyzer, "ngram_range": ngrams, "lowercase": False,
                "sublinear_tf": True, "min_df": min_df, "max_features": max_features or cap,
                "preprocessor": _PRE.get(name)}
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        parts.append((name, TfidfVectorizer(**opts)))
        weights[name] = w
    return FeatureUnion(parts, transformer_weights=weights)


def four_view_lr(max_features: int | None = None, C: float = 4.0, min_df: int = 2) -> Pipeline:
    return Pipeline([("feats", four_view_features(max_features, min_df)),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])
