import random

import pandas as pd
import pytest

from ntl.data import LABELS


@pytest.fixture
def toy_train():
    """12 prompts x 18 modelos; cada modelo tem um 'tique' de estilo detectável."""
    rng = random.Random(0)
    topics = ["física", "poesia", "biologia", "lógica", "história", "música",
              "química", "ética", "código", "astros", "mares", "cidades"]
    rows = []
    for p, topic in enumerate(topics):
        for k, m in enumerate(LABELS):
            words = [topic] * 5 + [rng.choice(["a", "b", "c", "d"]) for _ in range(20)]
            words.append(f"tique{k}")
            rows.append({"id": f"{p}-{k}", "prompt_id": f"p{p}", "text": " ".join(words), "model": m})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """Qwen3 minúsculo e aleatório + tokenizer de palavras, salvos como num Input do Kaggle.
    Só existe se o extra `nn` (torch/transformers) estiver instalado."""
    pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    words = ["<pad>", "<unk>", "<eos>", "a", "b", "c", "d"]
    words += [f"tique{k}" for k in range(18)]
    words += ["física", "poesia", "biologia", "lógica", "história", "música", "química", "ética",
              "código", "astros", "mares", "cidades"]
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>",
                                  unk_token="<unk>")
    d = tmp_path_factory.mktemp("models") / "qwen-3-embedding" / "tiny" / "1"
    tok.save_pretrained(d)
    cfg = Qwen3Config(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=128, pad_token_id=0)
    import torch

    torch.manual_seed(0)
    Qwen3Model(cfg).save_pretrained(d)
    return d
