import numpy as np
from sklearn.metrics import adjusted_rand_score

from ntl.groups import infer_groups


def test_infer_groups_recupera_os_prompts_pelo_tema(toy_train):
    g = infer_groups(toy_train["text"].tolist(), n_groups=toy_train["prompt_id"].nunique())
    assert len(g) == len(toy_train)
    assert adjusted_rand_score(toy_train["prompt_id"], g) > 0.95


def test_infer_groups_estima_numero_de_grupos(toy_train):
    g = infer_groups(toy_train["text"].tolist(), per_group=18)
    assert len(set(g)) == round(len(toy_train) / 18)


def test_infer_groups_com_um_ou_dois_textos():
    # no Kaggle, a sessão interativa tem test.csv com pouquíssimas linhas
    assert list(infer_groups(["só um texto"])) == [0]
    assert len(infer_groups(["texto um", "texto dois"])) == 2


# --- grupos com teto e limiar (não supõe n/18 grupos) ---
from ntl.groups import capped_groups


def test_capped_groups_recupera_prompts_sem_saber_quantos(toy_train):
    g = capped_groups(toy_train["text"].tolist(), cap=18)
    assert adjusted_rand_score(toy_train["prompt_id"], g) > 0.95


def test_capped_groups_nunca_passa_do_teto(toy_train):
    g = capped_groups(toy_train["text"].tolist(), cap=7)
    _, counts = np.unique(g, return_counts=True)
    assert counts.max() <= 7


def test_capped_groups_grupos_parciais_nao_sao_fundidos(toy_train):
    # teste desbalanceado: só alguns textos de cada prompt; n/18 juntaria prompts diferentes
    sub = toy_train.groupby("prompt_id").head(5)
    g = capped_groups(sub["text"].tolist(), cap=18)
    assert adjusted_rand_score(sub["prompt_id"], g) > 0.95


def test_capped_groups_textos_sem_relacao_ficam_sozinhos():
    g = capped_groups(["gato telhado miau", "planeta órbita sol", "bolsa ações juros"], cap=18)
    assert len(set(g)) == 3


def test_capped_groups_poucos_textos():
    assert list(capped_groups(["só um"])) == [0]
    assert list(capped_groups([])) == []
