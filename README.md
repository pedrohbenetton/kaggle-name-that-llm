# kaggle-name-that-llm

Atribuição de autoria entre LLMs na competição [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) do Kaggle: dado um parágrafo de 40 a 250 palavras, dizer qual de **18 modelos** (GPT, Gemini, Claude, Qwen, GLM, Kimi) o escreveu. Métrica: balanced accuracy (acaso = 0,056). O teste usa **prompts que não aparecem no treino**, e o treino tem só 2.699 textos (150 prompts × 18 modelos).

## Resultado

| Versão | O que é | CV agrupada | LB público |
|---|---|---|---|
| v1 | TF-IDF (palavras + caracteres) + regressão logística | 0,315 | 0,322 |
| v2 | 4 visões de estilo + atribuição em grupos de prompt | 0,480 | 0,345 |
| **v3** | 4 visões + embeddings congelados do Qwen3-Embedding-4B | 0,418 | **0,389** |
| v4 | v3 + Qwen3-Embedding-4B ajustado com LoRA | — | pendente |
| fork | notebook público de 0,683 (ensemble de terceiros, 30+ modelos) **+ invariantes tipográficas deste repo** | — | pendente |

**Contexto honesto da nota.** v1 a v3 são pipelines próprios, treinados só nos dados oficiais, em CPU e uma T4. Os notebooks públicos perto de 0,68 juntam dezenas de modelos de 4B a 27B, ajustados por outras pessoas com dados extras gerados por elas; o topo do leaderboard (0,76) usa métodos não publicados. A linha "fork" é o notebook público com uma única mudança minha (abaixo), medida separadamente, e o crédito é dos autores originais.

## O que este projeto mostra

**1. A validação certa importa mais que o modelo.** Cada prompt tem um texto de cada modelo. Com folds aleatórios o baseline cai para 0,104 (contra 0,315 agrupado por prompt): os outros 17 textos do mesmo prompt ficam no treino e o tema aponta para os modelos errados. Toda a avaliação aqui é GroupKFold por `prompt_id`.

**2. Um ganho de validação que não transferiu, e por quê.** Como cada prompt tem no máximo um texto por modelo, dá para escolher os rótulos de um grupo de prompt juntos (atribuição húngara). Na CV isso valia +0,10; no leaderboard, +0,002. Para entender, simulei o teste na validação (`ntl/sim.py`: cada modelo mantém 30% a 100% dos textos, como a página da competição avisa):

- a inferência de grupos por `n/18` cai de ARI 0,96 para 0,61 nesse cenário; agrupamento com teto de 18 e limiar de similaridade fica em 0,995 (`ntl/groups.py`);
- mesmo corrigido, o ganho cai de 0,48 para 0,435 na simulação e quase some no LB. Conclusão: o que sobe a nota é o modelo, não o pós-processamento. Os experimentos estão em `results/`.

**3. Estilo sem tema.** Duas visões de TF-IDF que ignoram o conteúdo (a "forma" das palavras, `Olá, Mundo` → `Xx, Xx`, e só as palavras funcionais) sobem a CV de 0,315 para 0,375 (`ntl/models.py`, ideia dos notebooks públicos, implementação própria).

**4. Embeddings de LLM se complementam com TF-IDF.** Sozinhos, os embeddings congelados do Qwen3-Embedding-4B dão 0,326; misturados às 4 visões (peso escolhido por CV), 0,418 de CV e 0,389 no LB, a primeira versão em que CV e LB andam juntos (`ntl/embed.py`, `ntl/blend.py`). A v4 ajusta o próprio modelo com LoRA, um fold de validação por GPU (`ntl/finetune.py`), testada de ponta a ponta num Qwen3 minúsculo em CPU.

**5. Invariantes tipográficas mineradas nos dados.** Alguns modelos nunca usam certas marcas nos 150 textos de cada um: travessão sem espaço (`—`) nunca aparece em gemini-3.1/3.6/3.8 (em 1.052 textos dos outros), apóstrofo reto nunca em gpt-5.6-sol (835 textos), aspas retas nunca em cinco modelos. `scripts/mine_invariants.py` acha essas marcas (só entram as com chance de acaso < 1e-4), `ntl/invariants.py` as aplica com reverificação no treino em tempo de execução, e `scripts/patch_public_notebook.py` aplica só isso ao notebook público de 0,683, que usava três dessas regras, uma delas incompleta.

## Estrutura

```
src/ntl/      data, cv, models (TF-IDF), groups, post, sim, embed, blend, finetune, invariants, submission
scripts/      experimentos, mineração de invariantes, gerador de notebooks e patch do notebook público
notebooks/    notebooks de submissão gerados a partir de src/ntl (autocontidos, sem internet)
results/      métricas dos experimentos
tests/        53 testes, incluindo os notebooks gerados rodando de ponta a ponta
```

## Como rodar

```bash
uv sync                      # extra opcional: uv sync --extra nn (torch CPU, para os testes de LoRA)
uv run pytest
uv run python scripts/run_cv.py --also-random   # baseline: CV agrupada e aleatória
uv run python scripts/exp_v2.py                 # 4 visões, grupos com teto, teste desbalanceado (~6 min)
uv run python scripts/mine_invariants.py        # invariantes tipográficas
uv run python scripts/build_kaggle_notebook.py --variant v3 --out notebooks/kaggle_v3.ipynb
```

Os dados da competição ficam em `name-that-llm/` e notebooks de terceiros em `notebooks/externos/`, ambos fora do git (as regras do Kaggle não permitem redistribuir os dados, e o código dos outros é deles).

## Limitações

- Uma execução por configuração, sem busca de hiperparâmetros; o limiar dos grupos e os pesos das misturas foram escolhidos nos mesmos folds em que são medidos (um pouco otimista).
- As invariantes foram mineradas em 150 textos por modelo; a reverificação em tempo de execução protege do treino, não de um comportamento novo no teste.
- O LB público usa ~40% do teste; a nota final é calculada nos outros 60%.

## Licença

MIT. O código compartilhado desta competição também precisa estar disponível no Kaggle (regra 6b da competição).
