"""Procura marcas tipográficas que algum modelo nunca usa no treino (invariantes negativas).
Para cada marca em >= 25 textos, P0 = chance de um modelo com a taxa dos demais nunca usá-la em
150 textos. Resultado usado em src/ntl/invariants.py (só entram P0 < 1e-4)."""

from __future__ import annotations

from ntl.data import load_train

df = load_train()
labs = sorted(df["model"].unique())
n = df["model"].value_counts()
cands = {c for t in df["text"] for c in t if not c.isalnum() and c != " "}
cands |= {" — ", " - ", " – ", "**", "\n\n", "\n- ", "...", "…", "'s", "’s", "n't", "e.g.", "i.e."}
rows = []
for m in sorted(cands):
    has = df["text"].str.contains(m, regex=False)
    if has.sum() < 25:
        continue
    cnt = df[has]["model"].value_counts().reindex(labs, fill_value=0)
    zero = [lab for lab in labs if cnt[lab] == 0]
    if not zero:
        continue
    ok = [lab for lab in labs if cnt[lab] > 0]
    rate = cnt[ok].sum() / n[ok].sum()
    rows.append(((1 - rate) ** 150, repr(m), int(has.sum()), zero))
for p0, m, k, zero in sorted(rows):
    print(f"P0={p0:.1e}  marca={m:8} textos={k:5}  nunca em: {', '.join(zero)}")
