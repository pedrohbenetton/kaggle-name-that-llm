"""Fine-tuning de um LLM de embeddings (Qwen3-Embedding) como classificador de 18 classes:
LoRA nas projeções de atenção e MLP, mean pooling da última camada e uma cabeça linear.
Na GPU (T4) usa fp16 com autocast e GradScaler; sem `peft`, treina só as 2 últimas camadas.
torch/transformers só são importados aqui dentro (o resto do pacote não precisa deles)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ntl.embed import masked_mean

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _build(model_dir, n_classes, device, use_lora, lora_r, grad_ckpt):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    from_cuda = str(device).startswith("cuda")
    dtype = torch.float16 if from_cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(model_dir, torch_dtype=dtype)
    base.config.use_cache = False
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            use_lora = False
    for p in base.parameters():
        p.requires_grad = False
    if use_lora:
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                         target_modules=LORA_TARGETS)
        base = get_peft_model(base, cfg)
        for p in base.parameters():  # adapters em fp32 (estável com GradScaler)
            if p.requires_grad:
                p.data = p.data.float()
    else:
        inner = base
        for mod in list(inner.layers[-2:]) + [inner.norm]:
            mod.float()
            for p in mod.parameters():
                p.requires_grad = True
    if grad_ckpt and from_cuda:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hidden = base.config.hidden_size
    head = nn.Linear(hidden, n_classes)
    return tok, base.to(device), head.to(device), use_lora


class TextClassifier:
    def __init__(self, tok, base, head, classes, max_len, device):
        self.tok, self.base, self.head = tok, base, head
        self.classes, self.max_len, self.device = list(classes), max_len, device

    def logits(self, texts: Sequence[str]):
        import torch

        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        cuda = str(self.device).startswith("cuda")
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            h = self.base(**enc).last_hidden_state
        pooled = masked_mean(h.float(), enc["attention_mask"])
        return self.head(pooled)


def train_classifier(
    texts: Sequence[str],
    labels: Sequence[str],
    model_dir: str | Path,
    *,
    epochs: int = 3,
    batch: int = 8,
    accum: int = 2,
    lr: float = 2e-4,
    head_lr: float = 1e-3,
    max_len: int = 384,
    lora_r: int = 16,
    use_lora: bool = True,
    grad_ckpt: bool = True,
    seed: int = 0,
    device: str | None = None,
    log: Callable[[str], object] | None = print,
) -> TextClassifier:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    classes = sorted(set(labels))
    y = torch.tensor([classes.index(v) for v in labels])
    tok, base, head, used_lora = _build(model_dir, len(classes), device, use_lora, lora_r,
                                        grad_ckpt)
    clf = TextClassifier(tok, base, head, classes, max_len, device)
    params = [p for p in base.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": lr},
                             {"params": head.parameters(), "lr": head_lr}], weight_decay=0.01)
    steps = epochs * math.ceil(len(texts) / (batch * accum))
    warm = max(1, steps // 10)

    def sched(s):
        return min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm))

    lrs = torch.optim.lr_scheduler.LambdaLR(opt, sched)
    cuda = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    rng = np.random.default_rng(seed)
    base.train()
    head.train()
    t0, skipped = time.time(), 0
    for ep in range(epochs):
        order = rng.permutation(len(texts))
        total, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b, s in enumerate(range(0, len(texts), batch)):
            idx = order[s : s + batch]
            out = clf.logits([texts[i] for i in idx])
            loss = loss_fn(out.float(), y[idx].to(device))
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss / accum).backward()
            total, n = total + float(loss.detach()), n + 1
            if (b + 1) % accum == 0 or s + batch >= len(texts):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                lrs.step()
        if log:
            log(f"[{device}] época {ep + 1}/{epochs} loss={total / max(n, 1):.4f} "
                f"lora={used_lora} pulados={skipped} {time.time() - t0:.0f}s")
    base.eval()
    head.eval()
    return clf


def predict_proba(clf: TextClassifier, texts: Sequence[str], batch: int = 16) -> np.ndarray:
    import torch

    order = np.argsort([len(t) for t in texts])
    out = np.empty((len(texts), len(clf.classes)), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            idx = order[s : s + batch]
            p = torch.softmax(clf.logits([texts[i] for i in idx]).float(), 1).cpu().numpy()
            out[idx] = p
    bad = ~np.isfinite(out).all(1)
    out[bad] = 1.0 / out.shape[1]  # fp16 estourou: previsão neutra
    return out
