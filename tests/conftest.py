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
