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
