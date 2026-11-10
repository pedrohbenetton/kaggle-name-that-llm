"""Aplica as invariantes tipográficas (src/ntl/invariants.py) a uma cópia local de um notebook
público da competição (ex.: o de 0,683), sem mexer em mais nada. O notebook público e o
resultado ficam em notebooks/externos/ (fora do git: o código é dos autores originais).

Mudança: um arquivo invariants.py a mais e, no ensemble.py, uma chamada depois do último
re-ranking e antes do argmax final, dentro de try/except (se falhar, sai igual ao original)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "    np.savez(work / 'final-logits.npz'"
CALL = '''    # --- Invariantes tipográficas mineradas no treino (pedrohbenetton/kaggle-name-that-llm) ---
    try:
        from invariants import apply_invariants, verified_rules
        _rules = verified_rules(official_texts,
                                [LABELS[i] for i in official_labels.to_numpy(dtype=np.int64)])
        logits, _changed = apply_invariants(logits, [str(t) for t in frame.text.values], LABELS,
                                            rules=_rules)
        print(f"Invariantes: {len(_rules)} regras verificadas no treino, "
              f"{_changed} previsões alteradas", flush=True)
    except Exception as _e_inv:
        print(f"Invariantes puladas: {_e_inv}", flush=True)

'''
NOTE = (
    "> **Cópia modificada por pedrohbenetton.** Única mudança: invariantes tipográficas "
    "negativas mineradas no train.csv (ex.: travessão sem espaço nunca aparece em "
    "gemini-3.1/3.6/3.8; apóstrofo reto nunca em gpt-5.6-sol; aspas retas nunca em "
    "gpt-5.6-luna/sol/terra, gpt-6-astra, qwen3.8-max, gemini-3.7), aplicadas antes do argmax "
    "final. Cada regra é reverificada no treino em tempo de execução. Todo o resto é do "
    "notebook original e dos autores creditados abaixo.\n\n"
)


def patch(nb: dict) -> dict:
    cells = nb["cells"]
    idx = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith(
        "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/ensemble.py"))
    src = "".join(cells[idx]["source"])
    assert src.count(ANCHOR) == 1, "ponto de inserção não encontrado"
    cells[idx]["source"] = src.replace(ANCHOR, CALL + ANCHOR).splitlines(True)
    module = (ROOT / "src" / "ntl" / "invariants.py").read_text(encoding="utf-8")
    module = module.replace("from __future__ import annotations\n", "")
    new = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
           "source": ("%%writefile /kaggle/temp/ntl-gsq-v27-runtime/invariants.py\n"
                      + module).splitlines(True)}
    cells.insert(idx, new)
    first = next(c for c in cells if c["cell_type"] == "markdown")
    first["source"] = (NOTE + "".join(first["source"])).splitlines(True)
    return nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    nb = patch(json.loads(Path(a.src).read_text(encoding="utf-8")))
    Path(a.out).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(a.out)
