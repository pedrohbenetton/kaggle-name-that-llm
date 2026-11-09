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

O acaso é 1/18 = 0,056.

### v2: visões de estilo e grupos com teto

Duas ideias tiradas dos notebooks públicos da competição (com implementação própria e medidas aqui):

- **4 visões de TF-IDF:** caracteres 3-5, palavras 1-3, a "forma" das palavras (`Olá, Mundo` → `Xx, Xx`) e só as palavras funcionais (o resto vira `CONTENT`). As duas últimas ignoram o tema.
- **Grupos com teto de 18 e limiar de similaridade**, em vez de forçar `n/18` grupos.

Para imitar o teste (onde os modelos não aparecem em número igual), cada fold de validação também é avaliado **desbalanceado**: cada modelo mantém de 30% a 100% dos seus textos (`ntl/sim.py`).

| Modelo | Pós-processamento | CV completo | CV desbalanceado |
|---|---|---|---|
| TF-IDF + LR | argmax | 0,315 | 0,320 |
| TF-IDF + LR | húngaro em n/18 grupos (LB 0,324) | 0,417 | 0,343 |
| TF-IDF + LR | húngaro em grupos com teto | 0,420 | 0,373 |
| 4 visões + LR | argmax | 0,375 | 0,376 |
| 4 visões + LR | centralização em grupos com teto | 0,424 | 0,422 |
| **4 visões + LR** | **húngaro em grupos com teto** | **0,480** | **0,435** |

Recuperação dos prompts (ARI): `n/18` cai de 0,96 para 0,61 no cenário desbalanceado; o teto com limiar 0,05 fica em 0,995. Isso explica parte do fracasso no leaderboard. O limiar foi escolhido nesses mesmos folds, então o número é um pouco otimista. Resultados em `results/exp_v2.json`.

**Leaderboard público:** TF-IDF + LR com atribuição em grupos inferidos **0,324**; o mesmo modelo só com argmax **0,322**. O baseline transferiu (CV 0,315 → LB 0,322), mas o ganho de +0,10 da atribuição **não apareceu no teste** (+0,002, dentro do ruído). Provável causa: o teste não tem a estrutura "um texto por modelo por prompt" (veja Riscos), então a restrição que funcionava no treino não vale lá. A v2 acima testa se o problema era a forma de inferir os grupos.

## O que aprendi com os dados

- **Cada prompt tem um texto de cada modelo** (150 prompts × 18 modelos). Isso tem duas consequências.
- **A validação aleatória engana para baixo, não para cima.** Com folds aleatórios o baseline cai para 0,104 (contra 0,315 agrupada): os outros 17 textos do mesmo prompt ficam no treino e o tema aponta para os modelos errados. A validação honesta para este teste (prompts novos) é a agrupada.
- **O grupo de prompt é uma restrição útil no treino (mas não no teste; veja o leaderboard).** Se cada prompt tem no máximo um texto por modelo, dá para escolher os rótulos do grupo juntos (atribuição húngara sobre as log-probabilidades), em vez de um por um.
- **Dá para recuperar os grupos sem rótulo.** Agrupamento hierárquico por cosseno no TF-IDF de palavras recupera os prompts com ARI 0,91, e o ganho fica a 0,3 ponto do teto.

## Riscos conhecidos

- A página diz que o número de textos por modelo no teste **não é igual** e que a mistura difere entre o leaderboard público e o privado. Se um prompt do teste tiver dois textos do mesmo modelo, a atribuição força um erro. O leaderboard vai dizer se a restrição vale.
- v1: o número de grupos no teste era estimado como `n/18` (a v2 usa teto + limiar).
- A simulação desbalanceada não cobre um prompt com dois textos do mesmo modelo, porque o treino não tem esse caso.
- Uma execução, sem ajuste de hiperparâmetros.

## Como rodar

```bash
uv sync
uv run pytest
uv run python scripts/run_cv.py --also-random     # baseline, CV agrupada e aleatória
uv run python scripts/exp_group_post.py           # centralização e atribuição por grupo
uv run python scripts/exp_v2.py                   # 4 visões, grupos com teto, teste desbalanceado (~6 min)
uv run python scripts/build_kaggle_notebook.py --variant v2_hungaro --out notebooks/kaggle_v2_hungaro.ipynb
uv run python scripts/build_kaggle_notebook.py    # gera notebooks/kaggle_submission.ipynb
```

Os dados da competição ficam em `name-that-llm/` (ignorado pelo git; as regras do Kaggle não permitem redistribuí-los). O notebook de submissão é gerado a partir de `src/ntl/` e roda sem internet; um teste executa o notebook gerado de ponta a ponta.
