import numpy as np
import pandas as pd
import pytest

from ntl.data import LABELS
from ntl.invariants import RULES, allowed_mask, apply_invariants, verified_rules

LAB = sorted(LABELS)


def test_travessao_exclui_tres_geminis():
    m = allowed_mask(["um texto—sem espaços em volta"], LAB, RULES)[0]
    assert not m[LAB.index("gemini-3.6-flash")] and m[LAB.index("gemini-3.7-flash")]


def test_sem_marcas_nada_muda():
    assert allowed_mask(["texto simples sem marcas"], LAB, RULES).all()


def test_contradicao_libera_a_linha():
    # regras que, juntas, não deixam nenhum rótulo: a linha fica sem máscara
    rules = [("A", set(LAB[:2])), ("B", set(LAB[2:4]))]
    assert allowed_mask(["A e B"], LAB, rules).all()


def test_apply_invariants_derruba_rotulo_proibido():
    logits = np.zeros((1, len(LAB)))
    logits[0, LAB.index("gpt-5.6-sol")] = 5.0  # o mais provável, mas a frase tem apóstrofo reto
    out, changed = apply_invariants(logits, ["it's here"], LAB)
    assert out[0].argmax() != LAB.index("gpt-5.6-sol") and changed == 1


def test_verified_rules_descarta_regra_violada():
    train = pd.DataFrame({"text": ["it's", "ok"], "model": ["gpt-5.6-sol", "kimi-k3"]})
    kept = verified_rules(train["text"], train["model"])
    assert "'" not in [m for m, _ in kept]


def test_regras_valem_no_treino_oficial():
    from ntl.data import load_train

    try:
        df = load_train()
    except FileNotFoundError:
        pytest.skip("sem dados locais")
    assert len(verified_rules(df["text"], df["model"])) == len(RULES)
    m = allowed_mask(df["text"].tolist(), LAB, RULES)
    assert m[np.arange(len(df)), [LAB.index(v) for v in df["model"]]].all()
