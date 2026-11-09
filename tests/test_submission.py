import pandas as pd
import pytest

from ntl.submission import make_submission


def test_submissao_valida():
    s = make_submission(["a", "b"], ["kimi-k3", "gpt-5.5"])
    assert list(s.columns) == ["id", "model"] and len(s) == 2


@pytest.mark.parametrize(
    "ids,preds",
    [(["a", "a"], ["kimi-k3", "gpt-5.5"]), (["a"], ["gpt-4"]), (["a"], [None]), (["a", "b"], ["kimi-k3"])],
)
def test_submissao_invalida(ids, preds):
    with pytest.raises(ValueError):
        make_submission(ids, preds)


def test_grava_parquet(tmp_path):
    from ntl.submission import write_submission

    p = write_submission(make_submission(["x"], ["glm-5.3"]), tmp_path)
    assert p.name == "submission.parquet"
    assert pd.read_parquet(p).to_dict("records") == [{"id": "x", "model": "glm-5.3"}]
