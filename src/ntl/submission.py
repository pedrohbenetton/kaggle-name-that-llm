"""Arquivo de submissão: as mesmas regras que o Kaggle usa para rejeitar."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ntl.data import LABELS


def make_submission(ids: Sequence, preds: Sequence) -> pd.DataFrame:
    if len(ids) != len(preds):
        raise ValueError("ids e previsões com tamanhos diferentes")
    s = pd.DataFrame({"id": list(ids), "model": list(preds)})
    if s.isna().any().any():
        raise ValueError("valores nulos")
    if s["id"].duplicated().any():
        raise ValueError("id repetido")
    fora = set(s["model"]) - set(LABELS)
    if fora:
        raise ValueError(f"rótulos inválidos: {sorted(fora)}")
    return s


def write_submission(s: pd.DataFrame, out_dir: Path | str = ".") -> Path:
    p = Path(out_dir) / "submission.parquet"
    s.to_parquet(p, index=False)
    return p
