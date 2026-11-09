import numpy as np

from ntl.cv import evaluate, grouped_folds, random_folds


def test_grouped_folds_nunca_dividem_um_prompt(toy_train):
    folds = grouped_folds(toy_train, n_splits=4)
    assert len(folds) == 4
    vistos = []
    for tr, va in folds:
        assert not set(toy_train.prompt_id.iloc[tr]) & set(toy_train.prompt_id.iloc[va])
        vistos.extend(va)
    assert sorted(vistos) == list(range(len(toy_train)))


def test_random_folds_misturam_prompts(toy_train):
    tr, va = random_folds(toy_train, n_splits=4, seed=0)[0]
    assert set(toy_train.prompt_id.iloc[tr]) & set(toy_train.prompt_id.iloc[va])


def test_evaluate_devolve_oof_e_balanced_accuracy(toy_train):
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import make_pipeline

    def factory():
        return make_pipeline(CountVectorizer(), DummyClassifier(strategy="most_frequent"))

    r = evaluate(factory, toy_train, grouped_folds(toy_train, 3))
    assert len(r["oof"]) == len(toy_train)
    assert abs(r["balanced_accuracy"] - 1 / 18) < 1e-9  # chuta sempre a mesma classe
    assert len(r["per_fold"]) == 3 and np.isfinite(r["std"])
    assert set(r["per_class_recall"]) == set(toy_train.model)
