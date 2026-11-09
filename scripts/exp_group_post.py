"""Experimento: quanto ganha usar o grupo de prompt no pós-processamento?
Usa o prompt_id verdadeiro (oráculo): é um teto, porque no teste o grupo precisa ser inferido."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ntl.cv import evaluate, grouped_folds
from ntl.data import load_train
from ntl.models import tfidf_lr
from ntl.post import assign_by_group, center_by_group

df = load_train()
r = evaluate(tfidf_lr, df, grouped_folds(df, 5))
P, classes, y, g = r["oof_proba"], np.array(r["classes"]), df["model"].to_numpy(), df["prompt_id"]
L = np.log(np.clip(P, 1e-9, 1))
variants = {
    "argmax": classes[P.argmax(1)],
    "centrado_por_prompt": classes[center_by_group(L, g).argmax(1)],
    "atribuicao_por_prompt": classes[assign_by_group(L, g)],
    "centrado+atribuicao": classes[assign_by_group(center_by_group(L, g), g)],
}
res = {k: round(float(balanced_accuracy_score(y, v)), 4) for k, v in variants.items()}
for k, v in res.items():
    print(f"{k:24} {v:.4f}")
Path("results").mkdir(exist_ok=True)
Path("results/exp_group_post.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
np.save("results/raw_oof_tfidf_lr.npy", P) if Path("results/raw").mkdir(exist_ok=True) is None else None
