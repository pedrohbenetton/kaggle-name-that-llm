"""Mede um modelo com CV agrupada por prompt (honesta) e, opcionalmente, aleatória (vazada)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ntl.cv import evaluate, grouped_folds, random_folds
from ntl.data import load_train
from ntl.models import tfidf_lr

MODELS = {"tfidf_lr": tfidf_lr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tfidf_lr", choices=list(MODELS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--also-random", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = load_train()
    schemes = {"grouped": grouped_folds(df, a.folds)}
    if a.also_random:
        schemes["random"] = random_folds(df, a.folds)
    out = {}
    for name, folds in schemes.items():
        t0 = time.perf_counter()
        r = evaluate(MODELS[a.model], df, folds)
        r.pop("oof")
        r["seconds"] = round(time.perf_counter() - t0, 1)
        out[name] = r
        print(f"{a.model} [{name}] balanced_acc={r['balanced_accuracy']:.4f} ±{r['std']:.4f} ({r['seconds']}s)")
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / f"cv_{a.model}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
