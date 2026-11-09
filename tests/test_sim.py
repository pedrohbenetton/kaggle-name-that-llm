import numpy as np

from ntl.sim import unbalanced_subsample


def test_subamostra_desbalanceia_modelos_e_grupos(toy_train):
    idx = unbalanced_subsample(toy_train, np.arange(len(toy_train)), seed=0, keep=(0.2, 1.0))
    sub = toy_train.iloc[idx]
    assert 0 < len(sub) < len(toy_train)
    counts = sub["model"].value_counts()
    assert counts.max() > counts.min()  # modelos com frequências diferentes
    assert sub.groupby("prompt_id").size().min() < 18  # grupos parciais


def test_subamostra_e_deterministica_e_dentro_do_indice(toy_train):
    va = np.arange(0, len(toy_train), 2)
    a = unbalanced_subsample(toy_train, va, seed=1)
    b = unbalanced_subsample(toy_train, va, seed=1)
    assert np.array_equal(a, b) and set(a) <= set(va)
