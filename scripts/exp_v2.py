"""Experimento v2: (a) modelo de 4 visões contra o baseline, (b) grupos com teto+limiar
contra n/18, (c) tudo isso na validação normal e num teste simulado desbalanceado."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score

from ntl.cv import evaluate, grouped_folds
from ntl.data import load_train
from ntl.groups import capped_groups, infer_groups
from ntl.models import four_view_lr, tfidf_lr
from ntl.post import assign_by_group, center_by_group
from ntl.sim import unbalanced_subsample

df = load_train()
folds = grouped_folds(df, 5)
y = df["model"].to_numpy()
texts = df["text"].tolist()
out: dict = {}

# (b) grupos: ARI por fold, completo e desbalanceado
ari: dict = {}
for name, fn in {
    "n/18": lambda t: infer_groups(t, per_group=18),
    **{f"teto thr={t}": partial(capped_groups, threshold=t) for t in (0.05, 0.1, 0.15, 0.2, 0.3)},
}.items():
    vals = {"completo": [], "desbalanceado": []}
    for k, (_, va) in enumerate(folds):
        for mode, idx in (("completo", va), ("desbalanceado", unbalanced_subsample(df, va, seed=k))):
            g = fn([texts[i] for i in idx])
            vals[mode].append(adjusted_rand_score(df["prompt_id"].to_numpy()[idx], g))
    ari[name] = {m: round(float(np.mean(v)), 3) for m, v in vals.items()}
    print("ARI", name, ari[name], flush=True)
out["ari"] = ari
best_thr = max((k for k in ari if k.startswith("teto")), key=lambda k: ari[k]["desbalanceado"])
thr = float(best_thr.split("=")[1])
out["thr_escolhido"] = thr

# (a) + (c) modelos e pós-processamento
for mname, factory in {"tfidf_lr": tfidf_lr, "four_view_lr": four_view_lr}.items():
    t0 = time.time()
    r = evaluate(factory, df, folds)
    classes = np.array(r["classes"])
    L = np.log(np.clip(r["oof_proba"], 1e-9, 1))
    res: dict = {"tempo_s": round(time.time() - t0)}
    for mode in ("completo", "desbalanceado"):
        rows, gi_n18, gi_cap = [], [], []
        for k, (_, va) in enumerate(folds):
            idx = va if mode == "completo" else unbalanced_subsample(df, va, seed=k)
            rows.append(idx)
            sub = [texts[i] for i in idx]
            gi_n18 += [f"{k}-{x}" for x in infer_groups(sub, per_group=18)]
            gi_cap += [f"{k}-{x}" for x in capped_groups(sub, threshold=thr)]
        idx = np.concatenate(rows)
        Ls, ys = L[idx], y[idx]
        def ba(p, ys=ys, classes=classes):
            return round(float(balanced_accuracy_score(ys, classes[p])), 4)

        res[mode] = {
            "argmax": ba(Ls.argmax(1)),
            "n18_centrado": ba(center_by_group(Ls, gi_n18).argmax(1)),
            "n18_hungaro": ba(assign_by_group(Ls, gi_n18)),
            "teto_centrado": ba(center_by_group(Ls, gi_cap).argmax(1)),
            "teto_hungaro": ba(assign_by_group(Ls, gi_cap)),
        }
    out[mname] = res
    print(mname, json.dumps(res), flush=True)
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    np.save(f"results/raw/oof_{mname}.npy", r["oof_proba"])

Path("results/exp_v2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
