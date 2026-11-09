# kaggle-name-that-llm

Solução para a competição [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) do Kaggle: dado um parágrafo de 40 a 250 palavras, dizer qual de 18 LLMs (OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot) o escreveu. Métrica: balanced accuracy. O teste usa **prompts que não aparecem no treino**.

## Resultados (validação cruzada local)

GroupKFold por `prompt_id` (5 folds), 2.699 textos de treino, balanced accuracy:

| Etapa | CV |
|---|---|
| TF-IDF (palavras 1-2 + caracteres 3-5) + regressão logística | 0,315 |
| + scores centralizados no grupo de prompt verdadeiro (teto) | 0,365 |
| + atribuição húngara no grupo de prompt verdadeiro (teto) | 0,420 |
| **+ atribuição húngara em grupos inferidos só pelo texto** | **0,417** |

Ainda sem nota no leaderboard. O acaso é 1/18 = 0,056.

## O que aprendi com os dados

- **Cada prompt tem um texto de cada modelo** (150 prompts × 18 modelos). Isso tem duas consequências.
- **A validação aleatória engana para baixo, não para cima.** Com folds aleatórios o baseline cai para 0,104 (contra 0,315 agrupada): os outros 17 textos do mesmo prompt ficam no treino e o tema aponta para os modelos errados. A validação honesta para este teste (prompts novos) é a agrupada.
- **O grupo de prompt é uma restrição útil.** Se cada prompt tem no máximo um texto por modelo, dá para escolher os rótulos do grupo juntos (atribuição húngara sobre as log-probabilidades), em vez de um por um.
- **Dá para recuperar os grupos sem rótulo.** Agrupamento hierárquico por cosseno no TF-IDF de palavras recupera os prompts com ARI 0,91, e o ganho fica a 0,3 ponto do teto.

## Riscos conhecidos

- A página diz que o número de textos por modelo no teste **não é igual** e que a mistura difere entre o leaderboard público e o privado. Se um prompt do teste tiver dois textos do mesmo modelo, a atribuição força um erro. O leaderboard vai dizer se a restrição vale.
- O número de grupos no teste é estimado como `n/18`.
- Uma execução, sem ajuste de hiperparâmetros.

## Como rodar

```bash
uv sync
uv run pytest
uv run python scripts/run_cv.py --also-random     # baseline, CV agrupada e aleatória
uv run python scripts/exp_group_post.py           # centralização e atribuição por grupo
uv run python scripts/build_kaggle_notebook.py    # gera notebooks/kaggle_submission.ipynb
```

Os dados da competição ficam em `name-that-llm/` (ignorado pelo git; as regras do Kaggle não permitem redistribuí-los). O notebook de submissão é gerado a partir de `src/ntl/` e roda sem internet; um teste executa o notebook gerado de ponta a ponta.
