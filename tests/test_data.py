import pandas as pd
import pytest

from ntl.data import LABELS, validate_train


def test_labels_oficiais():
    assert len(LABELS) == 18 and len(set(LABELS)) == 18
    assert {"gpt-6-astra", "claude-opus-5", "kimi-k3", "gemini-3.8-flash"} <= set(LABELS)


def test_validate_train_aceita_ok(toy_train):
    validate_train(toy_train)


@pytest.mark.parametrize("bad", ["coluna", "rotulo", "id_repetido", "texto_vazio"])
def test_validate_train_recusa(toy_train, bad):
    df = toy_train.copy()
    if bad == "coluna":
        df = df.drop(columns="prompt_id")
    elif bad == "rotulo":
        df.loc[0, "model"] = "gpt-4"
    elif bad == "id_repetido":
        df.loc[1, "id"] = df.loc[0, "id"]
    else:
        df.loc[0, "text"] = None
    with pytest.raises(ValueError):
        validate_train(pd.DataFrame(df))
