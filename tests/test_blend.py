import numpy as np

from ntl.blend import blend_logp, choose_weight, emb_lr
from ntl.cv import grouped_folds, oof_proba


def test_oof_proba_funciona_com_matriz(toy_train):
    rng = np.random.default_rng(0)
    y = toy_train["model"].to_numpy()
    classes = sorted(set(y))
    X = rng.normal(size=(len(y), 5)) + np.eye(18)[[classes.index(v) for v in y]] @ rng.normal(
        size=(18, 5)) * 5
    P, cls = oof_proba(emb_lr, X, y, grouped_folds(toy_train, 3))
    assert P.shape == (len(y), 18) and list(cls) == classes
    assert np.allclose(P.sum(1), 1)
    assert (np.array(cls)[P.argmax(1)] == y).mean() > 0.9


def test_blend_e_escolha_do_peso():
    y = np.array([0, 1, 0, 1])
    bom = np.log(np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]))
    ruim = np.log(np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]))
    assert np.allclose(blend_logp([bom, ruim], [1.0, 0.0]), bom)
    w, score = choose_weight(bom, ruim, y, np.array([0, 1]))
    assert w == 1.0 and score == 1.0
