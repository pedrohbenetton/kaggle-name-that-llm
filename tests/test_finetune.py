import numpy as np
import pytest

from ntl.blend import choose_weights
from ntl.cv import grouped_folds


def test_choose_weights_simplex():
    y = np.array([0, 1, 0, 1])
    bom = np.log(np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]))
    ruim = np.log(np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]))
    w, s = choose_weights([ruim, bom, ruim], y, np.array([0, 1]), step=0.5)
    assert s == 1.0 and w[1] == 1.0 and abs(sum(w) - 1) < 1e-9


def test_embed_texts_com_modelo_minusculo(tiny_model_dir, toy_train):
    from ntl.embed import embed_texts

    E = embed_texts(toy_train["text"].tolist()[:10], tiny_model_dir, fracs=(0.5, 1.0), batch=4)
    assert E.shape == (10, 64) and np.isfinite(E).all()


@pytest.fixture
def toy_forte(toy_train):
    """Mesmo toy, com o tique repetido: um modelo aleatório de 32 dimensões dilui um token só
    no mean pooling (nem uma LR sobre os embeddings congelados separa o toy original)."""
    df = toy_train.copy()
    df["text"] = [" ".join(t.split()[:5] + [t.split()[-1]] * 8 + t.split()[5:10])
                  for t in df["text"]]
    return df


@pytest.mark.parametrize("use_lora", [True, False])
def test_finetune_aprende_o_tique(tiny_model_dir, toy_forte, use_lora):
    from ntl.finetune import predict_proba, train_classifier

    tr, va = grouped_folds(toy_forte, 4)[0]
    X, y = toy_forte["text"].to_numpy(), toy_forte["model"].to_numpy()
    clf = train_classifier(list(X[tr]), list(y[tr]), tiny_model_dir, epochs=8, batch=16, accum=1,
                           lr=5e-3, head_lr=5e-2, max_len=64, use_lora=use_lora, log=None)
    P = predict_proba(clf, list(X[va]))
    assert P.shape == (len(va), 18) and np.allclose(P.sum(1), 1, atol=1e-4)
    assert list(clf.classes) == sorted(set(y))
    assert (np.asarray(clf.classes)[P.argmax(1)] == y[va]).mean() > 0.8
