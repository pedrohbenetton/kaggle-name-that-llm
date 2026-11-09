from ntl.cv import evaluate, grouped_folds
from ntl.models import tfidf_lr


def test_baseline_aprende_o_tique_de_estilo(toy_train):
    r = evaluate(lambda: tfidf_lr(max_word=2000, max_char=2000), toy_train, grouped_folds(toy_train, 3))
    assert r["balanced_accuracy"] > 0.9
