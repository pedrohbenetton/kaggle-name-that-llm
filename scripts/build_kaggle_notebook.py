"""Gera o notebook de submissão do Kaggle a partir de src/ntl (fonte única de verdade).
O notebook é autocontido: o código dos módulos é copiado, sem `import ntl`, porque a
competição roda sem internet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["data", "models", "groups", "post", "submission", "cv", "embed", "blend"]  # ordem de dependência

MAIN = '''
# Treina no train.csv inteiro, prevê o teste e aplica o pós-processamento da variante.
import numpy as np

train, test = load_train(), load_test()
model = __MODEL__()
model.fit(train["text"].to_numpy(), train["model"].to_numpy())
P = model.predict_proba(test["text"].fillna("").to_numpy())
classes = np.array(model.classes_)
__PRED__
path = write_submission(make_submission(test["id"].tolist(), pred.tolist()))
print(path, len(test), "linhas")
'''


MAIN_V3 = '''
# v3: 4 visões de TF-IDF + embeddings de um LLM congelado (Qwen3-Embedding, anexado em Input),
# misturados em log-probabilidade. O peso da mistura e o C do classificador de embeddings são
# escolhidos por CV agrupada por prompt aqui mesmo, sobre o treino (determinístico).
import os
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score

t0 = time.time()
train, test = load_train(), load_test()
Xtr, ytr = train["text"].to_numpy(), train["model"].to_numpy()
Xte = test["text"].fillna("").to_numpy()
folds = grouped_folds(train, 5)


def lg(P):
    return np.log(np.clip(P, 1e-9, 1))


def ba(P):
    return round(float(balanced_accuracy_score(ytr, classes[P.argmax(1)])), 4)


P1, classes = oof_proba(four_view_lr, Xtr, ytr, folds)
T1 = four_view_lr().fit(Xtr, ytr).predict_proba(Xte)
print("4 visões CV", ba(P1), f"{time.time() - t0:.0f}s", flush=True)

texts = [str(t) for t in Xtr] + [str(t) for t in Xte]
if os.environ.get("NTL_FAKE_EMBED"):
    E = fake_embed(texts)
else:
    mdir = find_model_dir("/kaggle/input", os.environ.get("NTL_EMB", "qwen-3-embedding"))
    assert mdir is not None, "anexe um modelo Qwen3-Embedding em Input (Add Input > Models)"
    print("modelo", mdir, flush=True)
    E = embed_texts(texts, mdir)
Etr, Ete = E[: len(Xtr)], E[len(Xtr) :]
print("embeddings", E.shape, f"{time.time() - t0:.0f}s", flush=True)

best = None
for C in (0.01, 0.05, 0.2):
    P2, c2 = oof_proba(lambda C=C: emb_lr(C), Etr, ytr, folds)
    assert list(c2) == list(classes)
    print(f"embeddings C={C} CV", ba(P2), flush=True)
    if best is None or ba(P2) > ba(best[1]):
        best = (C, P2)
C, P2 = best
T2 = emb_lr(C).fit(Etr, ytr).predict_proba(Ete)
w, s = choose_weight(lg(P2), lg(P1), ytr, classes)
print(f"mistura: peso embeddings={w} CV={s:.4f}", f"{time.time() - t0:.0f}s", flush=True)
pred = classes[blend_logp([lg(T2), lg(T1)], [w, 1 - w]).argmax(1)]
path = write_submission(make_submission(test["id"].tolist(), pred.tolist()))
print(path, len(test), "linhas")
'''

_LOGP = "L = np.log(np.clip(P, 1e-9, 1))\n"
_CAP = 'groups = capped_groups(test["text"].fillna("").tolist())\n'
# variante: (modelo, código da previsão, título)
VARIANTS = {
    "group": ("tfidf_lr", _LOGP + 'groups = infer_groups(test["text"].fillna("").tolist(), '
              'per_group=18)\npred = classes[assign_by_group(L, groups)]',
              "TF-IDF + LR, atribuição em n/18 grupos (LB 0,324)"),
    "argmax": ("tfidf_lr", "pred = classes[P.argmax(axis=1)]",
               "TF-IDF + LR, sem grupos (LB 0,322)"),
    "v2_argmax": ("four_view_lr", "pred = classes[P.argmax(axis=1)]",
                  "4 visões + LR, sem grupos"),
    "v2_centrado": ("four_view_lr", _LOGP + _CAP + "pred = classes[center_by_group(L, groups)"
                    ".argmax(axis=1)]", "4 visões + LR, centralização em grupos com teto"),
    "v2_hungaro": ("four_view_lr", _LOGP + _CAP + "pred = classes[assign_by_group(L, groups)]",
                   "4 visões + LR, atribuição em grupos com teto"),
    "v3": (None, None, "4 visões + embeddings Qwen3-Embedding (mistura escolhida por CV)"),
}


def module_source(name: str) -> str:
    src = (ROOT / "src" / "ntl" / f"{name}.py").read_text(encoding="utf-8")
    src = re.sub(r"^from ntl\.\S+ import .*\n", "", src, flags=re.MULTILINE)
    src = re.sub(r"^from __future__ import annotations\n", "", src, flags=re.MULTILINE)
    return f"# ---- ntl/{name}.py ----\n" + src


def build(variant: str = "group") -> dict:
    def code(s: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.strip("\n").splitlines(True)}

    model, pred, titulo = VARIANTS[variant]
    intro = (f"# Name That LLM: {titulo}\n\n"
             "Gerado por `scripts/build_kaggle_notebook.py` a partir de "
             "[kaggle-name-that-llm](https://github.com/pedrohbenetton/kaggle-name-that-llm). "
             "Validação: GroupKFold por prompt (balanced accuracy). As visões de estilo (forma, "
             "palavras funcionais) e o agrupamento com teto seguem ideias dos notebooks "
             "públicos da competição; a implementação é própria.")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(True)}]
    cells += [code(module_source(m)) for m in MODULES]
    main = MAIN_V3 if variant == "v3" else MAIN.replace("__MODEL__", model).replace("__PRED__", pred)
    cells += [code(main)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notebooks/kaggle_submission.ipynb")
    ap.add_argument("--variant", default="group", choices=list(VARIANTS))
    a = ap.parse_args()
    nb = build(a.variant)
    Path(a.out).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(a.out)
