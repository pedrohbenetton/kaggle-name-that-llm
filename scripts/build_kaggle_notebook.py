"""Gera o notebook de submissão do Kaggle a partir de src/ntl (fonte única de verdade).
O notebook é autocontido: o código dos módulos é copiado, sem `import ntl`, porque a
competição roda sem internet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["data", "models", "groups", "post", "submission"]  # ordem de dependência

MAIN = '''
# Treina no train.csv inteiro, prevê o teste, infere os grupos de prompt pelo conteúdo e
# impõe no máximo um texto por modelo em cada grupo (atribuição húngara).
import numpy as np

train, test = load_train(), load_test()
model = tfidf_lr()
model.fit(train["text"].to_numpy(), train["model"].to_numpy())
P = model.predict_proba(test["text"].fillna("").to_numpy())
classes = np.array(model.classes_)
__PRED__
path = write_submission(make_submission(test["id"].tolist(), pred.tolist()))
print(path, len(test), "linhas")
'''


PRED = {
    "group": (
        'groups = infer_groups(test["text"].fillna("").tolist(), per_group=18)\n'
        'pred = classes[assign_by_group(np.log(np.clip(P, 1e-9, 1)), groups)]'
    ),
    # diagnóstico: mesmo modelo, cada texto prevê sozinho (sem a restrição por grupo)
    "argmax": "pred = classes[P.argmax(axis=1)]",
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

    titulo = {"group": "+ atribuição por grupo de prompt", "argmax": "(diagnóstico: sem grupos)"}
    intro = (f"# Name That LLM: TF-IDF + regressão logística {titulo[variant]}\n\n"
             "Gerado por `scripts/build_kaggle_notebook.py` a partir de "
             "[kaggle-name-that-llm](https://github.com/pedrohbenetton/kaggle-name-that-llm). "
             "Validação: GroupKFold por prompt (balanced accuracy).")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(True)}]
    cells += [code(module_source(m)) for m in MODULES]
    cells += [code(MAIN.replace("__PRED__", PRED[variant]))]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notebooks/kaggle_submission.ipynb")
    ap.add_argument("--variant", default="group", choices=list(PRED))
    a = ap.parse_args()
    nb = build(a.variant)
    Path(a.out).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(a.out)
