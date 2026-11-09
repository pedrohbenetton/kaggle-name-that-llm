import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from ntl.data import LABELS


@pytest.mark.parametrize("variant", ["group", "argmax", "v2_argmax", "v2_centrado", "v2_hungaro"])
def test_notebook_gerado_roda_offline_e_gera_submissao_valida(tmp_path, toy_train, variant):
    data = tmp_path / "data"
    data.mkdir()
    toy_train.to_csv(data / "train.csv", index=False)
    test = toy_train.sample(40, random_state=0)[["id", "text"]]
    test.to_csv(data / "test.csv", index=False)
    nb = tmp_path / "submit.ipynb"
    subprocess.run([sys.executable, "scripts/build_kaggle_notebook.py", "--out", str(nb),
                    "--variant", variant], check=True)
    code = "\n\n".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"]
                       if c["cell_type"] == "code")
    env = {**os.environ, "NTL_DATA": str(data)}
    # `python -c` não define __file__, como num notebook
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
    sub = pd.read_parquet(tmp_path / "submission.parquet")
    assert list(sub.columns) == ["id", "model"]
    assert sorted(sub["id"]) == sorted(test["id"]) and set(sub["model"]) <= set(LABELS)
    assert "from ntl" not in code and "import ntl" not in code  # autocontido
    main = "".join(json.loads(nb.read_text())["cells"][-1]["source"])
    assert ("assign_by_group(" in main) == variant.endswith(("group", "hungaro"))
    assert ("four_view_lr()" in main) == variant.startswith("v2")
