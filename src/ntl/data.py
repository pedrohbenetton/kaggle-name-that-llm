"""Rótulos oficiais, caminhos e validação dos dados da competição."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

LABELS: tuple[str, ...] = (
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
)

# No Kaggle: /kaggle/input/... Local: pasta baixada dentro do projeto (ignorada pelo git).
# Num notebook não existe __file__, por isso o caminho local é opcional.
_LOCAL = (
    str(Path(globals()["__file__"]).resolve().parents[2] / "name-that-llm" / "name-that-llm")
    if "__file__" in globals()
    else ""
)
_CANDIDATES = (
    os.environ.get("NTL_DATA", ""),
    "/kaggle/input/competitions/name-that-llm",
    "/kaggle/input/name-that-llm",
    _LOCAL,
)


def data_dir() -> Path:
    for c in _CANDIDATES:
        if c and (Path(c) / "train.csv").exists():
            return Path(c)
    raise FileNotFoundError("train.csv não encontrado; defina NTL_DATA com a pasta dos dados")


def validate_train(df: pd.DataFrame) -> None:
    falta = {"id", "prompt_id", "text", "model"} - set(df.columns)
    if falta:
        raise ValueError(f"colunas ausentes: {sorted(falta)}")
    if df["id"].duplicated().any():
        raise ValueError("id repetido")
    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        raise ValueError("texto vazio")
    fora = set(df["model"]) - set(LABELS)
    if fora:
        raise ValueError(f"rótulos fora da lista oficial: {sorted(fora)}")


def load_train(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv((path or data_dir()) / "train.csv")
    validate_train(df)
    return df


def load_test(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv((path or data_dir()) / "test.csv")
