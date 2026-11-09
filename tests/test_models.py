from ntl.cv import evaluate, grouped_folds
from ntl.models import tfidf_lr


def test_baseline_aprende_o_tique_de_estilo(toy_train):
    r = evaluate(lambda: tfidf_lr(max_word=2000, max_char=2000), toy_train, grouped_folds(toy_train, 3))
    assert r["balanced_accuracy"] > 0.9


def test_quatro_visoes_aprende_o_tique_de_estilo(toy_train):
    from ntl.models import four_view_lr

    r = evaluate(lambda: four_view_lr(max_features=2000), toy_train, grouped_folds(toy_train, 3))
    assert r["balanced_accuracy"] > 0.9


def test_visoes_forma_e_funcao():
    from ntl.models import function_view, shape_view

    assert shape_view("Olá, Mundo 42!") == "Xx, Xx 42!"
    assert function_view("The cat sat on the mat.") == "the CONTENT CONTENT on the CONTENT ."
