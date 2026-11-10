"""Embeddings de um LLM congelado (mean pooling de algumas camadas), para usar como features.
No Kaggle o modelo vem como Input (sem internet); torch/transformers só são importados ao
extrair, então o resto do pacote roda sem eles."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def masked_mean(h, mask):
    """Média dos estados ocultos só nos tokens reais. h: (b, t, d), mask: (b, t).
    Funciona com numpy e com tensores torch."""
    m = mask[..., None] * 1.0
    return (h * m).sum(1) / m.sum(1)


def layer_indices(n_layers: int, fracs: Sequence[float]) -> list[int]:
    return sorted({min(n_layers, max(1, round(f * n_layers))) for f in fracs})


def find_model_dir(root: str | Path, pattern: str) -> Path | None:
    """Primeira pasta sob `root` com config.json cujo caminho contém `pattern`."""
    for cfg in sorted(Path(root).rglob("config.json")):
        if pattern in str(cfg.parent):
            return cfg.parent
    return None


def embed_texts(
    texts: Sequence[str],
    model_dir: str | Path,
    fracs: Sequence[float] = (0.5, 0.75, 1.0),
    max_len: int = 512,
    batch: int = 16,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_dir, torch_dtype=dtype).to(dev).eval()
    idx = layer_indices(model.config.num_hidden_layers, fracs)
    order = np.argsort([len(t) for t in texts])  # lotes de tamanho parecido: menos padding
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            chunk = [texts[i] for i in order[s : s + batch]]
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states
            feats = [masked_mean(hs[i].float(), enc["attention_mask"]) for i in idx]
            out.append(torch.cat(feats, 1).cpu().numpy())
    E = np.empty((len(texts), out[0].shape[1]), dtype=np.float32)
    E[order] = np.concatenate(out)
    return E


def fake_embed(texts: Sequence[str], dim: int = 64) -> np.ndarray:
    """Só para testar o notebook sem GPU: SVD de TF-IDF de caracteres."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(analyzer="char", ngram_range=(2, 4)).fit_transform(texts)
    return TruncatedSVD(min(dim, X.shape[1] - 1), random_state=0).fit_transform(X)
