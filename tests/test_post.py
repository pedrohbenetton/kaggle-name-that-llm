import numpy as np

from ntl.post import assign_by_group, center_by_group


def test_center_remove_media_do_grupo():
    s = np.array([[1.0, 5.0], [3.0, 7.0], [10.0, 0.0]])
    c = center_by_group(s, ["g", "g", "h"])
    assert np.allclose(c[:2].mean(axis=0), 0) and np.allclose(c[2], 0)


def test_assign_um_modelo_por_texto_dentro_do_grupo():
    # os dois textos do grupo preferem a classe 0; a atribuição dá a classe 1 ao menos confiante
    s = np.array([[0.9, 0.8], [0.6, 0.1], [0.2, 0.7]])
    pred = assign_by_group(s, ["g", "g", "h"])
    assert list(pred[:2]) == [1, 0] and pred[2] == 1


def test_assign_grupo_maior_que_classes_cai_no_argmax():
    s = np.array([[0.9, 0.1]] * 3)
    assert list(assign_by_group(s, ["g"] * 3)) == [0, 0, 0]
