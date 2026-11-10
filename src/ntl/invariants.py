"""Invariantes tipográficas negativas: marcas que certos modelos NUNCA usam no treino
(2.699 textos, 150 por modelo). Se o texto tem a marca, esses rótulos saem da disputa.

Mineradas em `scripts/mine_invariants.py`: só entram marcas em >= 25 textos cuja ausência em
um modelo seria improvável por acaso (P < 1e-4 se ele usasse a marca na taxa dos demais).
As três primeiras já existiam no notebook público de 0,683 (parcialmente); as outras são novas."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_ALL = {
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
}
_CLAUDE_GLM = {m for m in _ALL if m.startswith(("claude", "glm"))}

# (marca, rótulos permitidos quando a marca aparece)
RULES: list[tuple[str, set[str]]] = [
    (" — ", {"claude-opus-4-6", "claude-opus-5", "glm-5.3", "glm-5.3-flash"}),
    ("’", _ALL - _CLAUDE_GLM),
    ("“", _ALL - _CLAUDE_GLM - {"gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash"}),
    ("”", _ALL - _CLAUDE_GLM - {"gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash"}),
    ("—", _ALL - {"gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.8-flash"}),
    ("'", _ALL - {"gpt-5.6-sol"}),
    ('"', _ALL - {"gemini-3.7-flash", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
                  "gpt-6-astra", "qwen3.8-max"}),
]


def verified_rules(texts: Sequence[str], labels: Sequence[str], rules=None):
    """Mantém só as regras sem nenhuma violação nos dados dados (checagem de segurança)."""
    rules = RULES if rules is None else rules
    texts, labels = [str(t) for t in texts], list(labels)
    return [(m, ok) for m, ok in rules
            if all(lab in ok for t, lab in zip(texts, labels, strict=True) if m in t)]


def allowed_mask(texts: Sequence[str], labels: Sequence[str], rules=None) -> np.ndarray:
    rules = RULES if rules is None else rules
    labels = list(labels)
    mask = np.ones((len(texts), len(labels)), dtype=bool)
    for m, ok in rules:
        col = np.array([lab in ok for lab in labels])
        hit = np.array([m in str(t) for t in texts], dtype=bool)
        mask[hit] &= col
    mask[~mask.any(1)] = True  # regras em contradição: não mexe na linha
    return mask


def apply_invariants(logits: np.ndarray, texts: Sequence[str], labels: Sequence[str],
                     rules=None, penalty: float = 30.0) -> tuple[np.ndarray, int]:
    """Subtrai `penalty` dos rótulos proibidos. Devolve os logits e quantas previsões mudaram."""
    mask = allowed_mask(texts, labels, rules)
    out = np.asarray(logits, dtype=np.float64) - penalty * (~mask)
    return out, int((out.argmax(1) != np.asarray(logits).argmax(1)).sum())
