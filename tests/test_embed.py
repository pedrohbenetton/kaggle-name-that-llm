import numpy as np

from ntl.embed import find_model_dir, layer_indices, masked_mean


def test_masked_mean_ignora_padding():
    h = np.array([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])  # 1 texto, 3 tokens, dim 2
    m = np.array([[1, 1, 0]])
    assert np.allclose(masked_mean(h, m), [[2.0, 2.0]])


def test_layer_indices_por_fracao():
    assert layer_indices(28, (0.5, 0.75, 1.0)) == [14, 21, 28]
    assert layer_indices(4, (0.1, 1.0)) == [1, 4]  # nunca a camada 0 (embeddings de token)


def test_find_model_dir_acha_config(tmp_path):
    d = tmp_path / "models" / "qwen-lm" / "qwen-3-embedding" / "transformers" / "4b" / "1"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (tmp_path / "outro").mkdir()
    assert find_model_dir(tmp_path, "qwen-3-embedding") == d
    assert find_model_dir(tmp_path, "nao-existe") is None
