# GraphIDS: detecção de intrusão em rede com grafos e aprendizado auto-supervisionado

Reimplementação, feita do zero em PyTorch, do artigo **"Self-Supervised Learning of
Graph Representations for Network Intrusion Detection"** (Guerra et al., NeurIPS 2025),
com reprodução dos resultados nos quatro datasets do artigo e avaliação num dataset
novo, o **NF-ToN-IoT-v3**. Projeto da disciplina CIN0114 (Técnicas de Ataque e
Detecção de Intrusão, CIn-UFPE, 2026.2).

[![CI](https://github.com/rxavier1904/projeto-GraphIDS-equipe-10/actions/workflows/ci.yml/badge.svg)](https://github.com/rxavier1904/projeto-GraphIDS-equipe-10/actions/workflows/ci.yml)

## Sumário

- [O problema e a ideia do artigo](#o-problema-e-a-ideia-do-artigo)
- [Como o modelo funciona](#como-o-modelo-funciona)
- [Datasets](#datasets)
- [Resultados](#resultados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Como rodar](#como-rodar)
- [Configuração dos experimentos](#configuração-dos-experimentos)
- [Testes e qualidade de código](#testes-e-qualidade-de-código)
- [Documentação complementar](#documentação-complementar)
- [Equipe](#equipe)
- [Referências](#referências)

## O problema e a ideia do artigo

Sistemas de detecção de intrusão (IDS) supervisionados aprendem a reconhecer os
ataques que aparecem no treino e tendem a falhar diante de ataques novos, além de
dependerem de dados rotulados, que são caros de obter. O GraphIDS segue outro caminho:
aprende **apenas o comportamento normal** da rede, sem nenhum rótulo de ataque, e
marca como suspeito tudo o que foge desse padrão (detecção de anomalias).

Dois pontos distinguem o método:

1. **O tráfego vira um grafo.** Cada host (endereço IP) é um nó e cada fluxo NetFlow é
   uma aresta dirigida da origem para o destino, com as features do fluxo na aresta.
   Assim, o modelo enxerga cada fluxo junto com o contexto dos hosts que ele liga, o
   que ajuda a perceber padrões como varreduras e ataques distribuídos.
2. **Representação e detecção são treinadas juntas.** Um GNN (E-GraphSAGE) gera um
   embedding por fluxo e um autoencoder Transformer aprende a reconstruir esses
   embeddings. O erro de reconstrução é o score de anomalia, e o gradiente dele
   atravessa os dois componentes, de modo que o GNN aprende embeddings úteis para a
   detecção.

## Como o modelo funciona

```
CSV NetFlow ─► limpeza ─► split 80/10/10 ─► Min-Max (ajustado no treino benigno)
                                                   │
                                                   ▼
                              grafo dirigido: hosts = nós, fluxos = arestas
                                                   │
                                                   ▼
          E-GraphSAGE (1 hop) ─► embedding h_i de cada fluxo
                                                   │
                                                   ▼
          janelas de até 512 fluxos ─► Transformer encoder-decoder com máscara
                                       de atenção ─► reconstrução ĥ_i
                                                   │
                                                   ▼
          score s_i = ‖h_i − ĥ_i‖²  ─►  ataque se s_i > threshold
```

**E-GraphSAGE** ([`graphids/models/egraphsage.py`](graphids/models/egraphsage.py)).
Para um fluxo `u → v`, cada host `w` recebe a média das features dos fluxos que chegam
nele; essa média passa por uma camada linear com ReLU e gera o embedding do host. O
embedding do fluxo é a concatenação `[h_u ‖ h_v]` projetada por outra camada linear,
com dropout. A amostragem de vizinhança (até `fanout` arestas por nó) é implementada
por nós em [`graphids/models/sampling.py`](graphids/models/sampling.py), sem PyTorch
Geometric: como a média é feita sobre as features de entrada, ela é pré-calculada e só
reamostrada para os nós com grau acima do `fanout`.

**Autoencoder Transformer com máscara** ([`graphids/models/transformer_mae.py`](graphids/models/transformer_mae.py)).
Os embeddings são projetados para uma dimensão menor (`embed_dim`) e agrupados em
janelas de `window_size` fluxos, sem positional encoding (os fluxos são embaralhados).
O encoder gera representações contextuais e o decoder reconstrói a sequência com
cross-attention sobre a saída do encoder. No treino, uma máscara simétrica desliga uma
fração `mask_ratio` das ligações de atenção (um dropout estruturado, sem remover
tokens); na inferência a máscara fica desligada.

**Treino** ([`graphids/training/trainer.py`](graphids/training/trainer.py)). O treino
usa só fluxos benignos e minimiza a média do erro de reconstrução, com um único
backward que atualiza GNN e Transformer juntos. O otimizador é AdamW com weight decay
separado para cada componente, e há early stopping pela PR-AUC de validação.

**Threshold** ([`graphids/training/threshold.py`](graphids/training/threshold.py)).
São testados 500 thresholds igualmente espaçados entre o menor e o maior score da
validação, e fica o que maximiza o F1 macro. O mesmo valor é aplicado ao teste, sem
olhar para ele.

**Métricas** ([`graphids/evaluation/metrics.py`](graphids/evaluation/metrics.py)).
PR-AUC (average precision, como no artigo), F1 macro e taxa de detecção por categoria
de ataque.

## Datasets

Todos vêm da coleção NetFlow da Universidade de Queensland
(<https://staff.itee.uq.edu.au/marius/NIDS_datasets/>) e compartilham o mesmo formato
de features NetFlow, então o mesmo pipeline serve para todos.

| Dataset | Papel | Fluxos | Hosts | Ataques | Categorias de ataque | Amostra usada |
|---|---|---:|---:|---:|---:|---|
| NF-UNSW-NB15-v2 | artigo | 2.390.275 | 44 | 3,98% | 9 | completo |
| NF-UNSW-NB15-v3 | artigo | 2.365.424 | 44 | 5,40% | 9 | completo |
| NF-CSE-CIC-IDS2018-v2 | artigo | 18.893.708 | 255.042 | 11,95% | 14 | 20% estratificado |
| NF-CSE-CIC-IDS2018-v3 | artigo | 20.115.529 | 205.801 | 12,93% | 14 | 20% estratificado |
| **NF-ToN-IoT-v3** | **novo** | 27.520.260 | 19.555 | 38,98% | 9 | 20% estratificado |

Os números foram calculados pelo nosso pipeline
([`results/estatisticas/resumo_datasets.csv`](results/estatisticas/resumo_datasets.csv))
e, nos quatro datasets do artigo, batem exatamente com a Tabela 1 dele. A contagem de
fluxos por categoria em treino, validação e teste está em
[`results/estatisticas/`](results/estatisticas/).

**Por que o NF-ToN-IoT-v3.** Tem as mesmas features NetFlow v3 dos datasets do artigo,
então o pipeline roda sem adaptação, mas vem de um ambiente IoT/IIoT, diferente dos
dois do artigo, e traz ataques que o artigo não cobre (ransomware, injection, XSS,
MITM, backdoor, password). A proporção de ataques é alta (cerca de 39%), o que muda a
leitura da PR-AUC: a linha de base de um classificador aleatório é a própria
prevalência. Como o dataset tem escala parecida com a do CIC-IDS2018, usamos a mesma
amostra estratificada de 20%.

**Pré-processamento** ([`graphids/data/`](graphids/data/)), seguindo a Seção 4.2 do
artigo:

1. leitura do CSV, troca de ±inf e NaN por 0, separação de `Label` e `Attack`;
2. remoção dos IPs das features (eles viram os nós do grafo) e, nos datasets v3, dos
   timestamps;
3. amostra estratificada opcional (`fraction`), feita antes do split;
4. split estratificado por categoria de ataque: 80% treino, 10% validação, 10% teste;
   os ataques são removidos do treino, que fica só com fluxos benignos;
5. normalização Min-Max ajustada no treino e aplicada à validação e ao teste, com clip
   em [−10, 10];
6. construção de três grafos separados, com o mapa IP → nó compartilhado entre eles;
7. cache dos grafos em `data/cache/`, para as execuções seguintes não refazerem o
   processamento.

## Resultados

Todas as execuções usaram as seeds 1, 2 e 3, os hiperparâmetros da Tabela 14 do artigo
e GPUs NVIDIA A100 do cluster SLURM do CIn. Os valores abaixo são média ± desvio entre
as seeds, no conjunto de teste, e saem de
[`results/tabela_reproducao.csv`](results/tabela_reproducao.csv).

### Reprodução (comparação com a Tabela 3 do artigo)

| Dataset | PR-AUC (nosso) | PR-AUC (artigo) | F1 macro (nosso) | F1 macro (artigo) |
|---|---|---|---|---|
| NF-UNSW-NB15-v3 | 0,9983 ± 0,0029 | 0,9998 ± 0,0007 | 0,9999 ± 0,0001 | 0,9961 ± 0,0084 |
| NF-CSE-CIC-IDS2018-v3 | 0,8714 ± 0,0531 | 0,8819 ± 0,0347 | 0,9252 ± 0,0435 | 0,9447 ± 0,0213 |
| NF-UNSW-NB15-v2 | 0,7678 ± 0,1236 | 0,8116 ± 0,0367 | 0,9232 ± 0,0038 | 0,9264 ± 0,0217 |
| NF-CSE-CIC-IDS2018-v2 | 0,9344 ± 0,0207 | 0,9201 ± 0,0238 | 0,9567 ± 0,0179 | 0,9431 ± 0,0131 |

O critério de reprodução combinado no [plano](docs/PLANO.md) é a média ficar a até 2
desvios do valor publicado. Ele é atendido em 7 das 8 métricas. A exceção é a PR-AUC
do NF-UNSW-NB15-v3, que fica só 0,0015 abaixo do artigo, mas o desvio publicado é muito
pequeno (0,0007); o F1 macro do mesmo dataset ficou acima do artigo.

### Novo dataset: NF-ToN-IoT-v3

| Dataset | PR-AUC | F1 macro |
|---|---|---|
| NF-ToN-IoT-v3 | 0,6872 ± 0,0192 | 0,6408 ± 0,1488 |

O desempenho é bem menor que nos datasets do artigo (a prevalência de ataques, que é a
linha de base da PR-AUC, é de cerca de 39%). Pela taxa de detecção por categoria
([`results/figuras/deteccao_por_ataque.csv`](results/figuras/deteccao_por_ataque.csv)),
backdoor e ransomware são detectados em 100% dos fluxos de teste, enquanto scanning
(cerca de 4%), MITM, DoS, XSS, password e DDoS ficam bem abaixo; a taxa de falso
positivo nos benignos é de cerca de 10%.

### Detecção por categoria nos datasets do artigo

No NF-UNSW-NB15-v3 todas as categorias de ataque são detectadas em 100% dos fluxos de
teste, com taxa de falso positivo de cerca de 0,003%. No NF-CSE-CIC-IDS2018 as
categorias mais difíceis são Infilteration (7% no v2 e 27% no v3) e Bot, cuja detecção
varia muito entre seeds; as demais ficam perto de 100%.

### Custo computacional

Fonte: [`results/tabela_custo.csv`](results/tabela_custo.csv).

| Dataset | Parâmetros | Tempo de treino (min) | Inferência (µs/fluxo) | Pico de memória da GPU (MB) |
|---|---:|---|---|---:|
| NF-UNSW-NB15-v3 | 99.778 | 8,7 ± 3,5 | 0,76 ± 0,57 | 901 |
| NF-CSE-CIC-IDS2018-v3 | 59.314 | 8,7 ± 4,1 | 0,44 ± 0,00 | 1.201 |
| NF-UNSW-NB15-v2 | 93.194 | 2,5 ± 0,9 | 0,24 ± 0,00 | 1.260 |
| NF-CSE-CIC-IDS2018-v2 | 57.562 | 7,8 ± 6,5 | 0,69 ± 0,20 | 1.068 |
| NF-ToN-IoT-v3 | 99.778 | 5,4 ± 2,7 | 0,82 ± 0,24 | 1.175 |

### Figuras

Em [`results/figuras/`](results/figuras/), uma série por dataset, em PDF para o
relatório:

- `<dataset>_pr.pdf`: curva precisão-revocação no teste, uma linha por seed (Figura 4
  do artigo);
- `<dataset>_scores.pdf`: distribuição do score de anomalia por categoria, com o
  threshold (Figuras 5 a 8);
- `<dataset>_deteccao.pdf`: taxa de detecção por categoria de ataque;
- `<dataset>_treino.pdf`: perdas e PR-AUC de validação por época;
- `<dataset>_tsne.pdf`: t-SNE das features, dos embeddings do E-GraphSAGE e da saída
  do encoder do Transformer (Figuras 9 a 12).

Os artefatos de cada execução (`metrics.json`, `history.json` e `config.yaml`) estão em
[`results/runs/<dataset>/seed<s>/`](results/runs/).

## Estrutura do repositório

```
graphids/                 pacote Python
  config.py               configuração de um experimento (YAML + sobrescritas)
  cli.py                  ponto de entrada: python -m graphids.cli
  data/                   leitura, limpeza, split, normalização, grafos, cache, estatísticas
  models/                 E-GraphSAGE, amostragem de vizinhança, Transformer MAE, GraphIDS
  training/               laço de treino, early stopping, threshold, execução completa
  evaluation/             métricas, figuras e t-SNE dos embeddings
configs/                  um YAML por dataset (hiperparâmetros da Tabela 14) + sintético
scripts/                  download dos datasets, jobs SLURM, várias seeds, tabelas e figuras
notebooks/                estatísticas dos datasets
results/                  métricas por execução, tabelas agregadas, estatísticas e figuras
tests/                    testes com dados sintéticos (rodam em CPU, em segundos)
docs/                     plano, contratos entre módulos, servidor, notas de referência
```

## Instalação

Requer Python 3.10 ou mais recente. O projeto usa só PyTorch, sem PyTorch Geometric.

```bash
git clone https://github.com/rxavier1904/projeto-GraphIDS-equipe-10.git
cd projeto-GraphIDS-equipe-10
python3 -m venv .venv
source .venv/bin/activate

# PyTorch compatível com o seu CUDA (ou a versão CPU, para os testes)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # ajuste cu1xx
pip install -e ".[dev]"

python scripts/check_env.py --cpu-ok   # confere versões e GPU
pytest -q                              # testes sintéticos
```

Dependências principais: `torch`, `numpy`, `pandas`, `scikit-learn`, `pyyaml` e
`matplotlib` (ver [`pyproject.toml`](pyproject.toml)).

## Como rodar

### 1. Teste rápido com dados sintéticos

Não precisa de dataset nem de GPU:

```bash
python -m graphids.cli --config configs/synthetic.yaml
```

### 2. Obter os datasets

Os links da UQ abrem uma página de download, não o arquivo. Baixe o zip de cada dataset
pelo navegador, coloque em `data/raw/` e rode:

```bash
scripts/download_datasets.sh data                    # os 5 datasets
scripts/download_datasets.sh data NF-UNSW-NB15-v3    # só um
```

O script descompacta, organiza os CSVs em `data/<dataset>/<dataset>.csv` e registra
tamanho e SHA-256 em `data/MANIFEST.tsv`. Também aceita um link direto do zip pela
variável `URL_<nome>` (ex.: `URL_NF_UNSW_NB15_v3=<link>`).

### 3. Pré-processamento e estatísticas

O pré-processamento acontece automaticamente na primeira execução de cada configuração
e fica em cache em `data/cache/`. Para gerar antes o cache e as tabelas de estatísticas
(`results/estatisticas/`):

```bash
python -m graphids.data.stats --seed 1 \
    --config configs/NF-UNSW-NB15-v3.yaml --config configs/NF-ToN-IoT-v3.yaml
```

O notebook [`notebooks/estatisticas_datasets.ipynb`](notebooks/estatisticas_datasets.ipynb)
faz o mesmo de forma interativa.

### 4. Treino e avaliação

```bash
# uma seed
python -m graphids.cli --config configs/NF-UNSW-NB15-v3.yaml --seed 1 --data-dir data

# só avaliar um checkpoint já treinado
python -m graphids.cli --config configs/NF-UNSW-NB15-v3.yaml --seed 1 --test

# sobrescrever hiperparâmetros sem editar o YAML
python -m graphids.cli --config configs/NF-UNSW-NB15-v3.yaml --set num_epochs=5

# várias seeds em sequência (seeds já concluídas são puladas)
scripts/run_seeds.sh configs/NF-UNSW-NB15-v3.yaml 1 2 3
```

Opções da CLI: `--config`, `--seed`, `--data-dir`, `--output-dir`, `--device`
(`auto`, `cpu` ou `cuda`), `--test`, `--checkpoint` e `--set CHAVE=VALOR` (repetível).

Cada execução grava em `results/runs/<dataset>/seed<s>/`:

| Arquivo | Conteúdo |
|---|---|
| `metrics.json` | PR-AUC e F1 macro no teste, melhor PR-AUC de validação, threshold, época, tempo, memória, nº de parâmetros, commit |
| `history.json` | perdas e métricas de validação por época |
| `config.yaml` | configuração efetiva usada |
| `scores_val.npz`, `scores_test.npz` | scores, rótulos e categorias de cada fluxo (fora do git) |
| `model.ckpt` | melhor checkpoint (fora do git) |

### 5. Cluster SLURM

```bash
scripts/jobs/submit_all.sh                              # 4 datasets do artigo × 3 seeds
scripts/jobs/submit_all.sh configs/NF-ToN-IoT-v3.yaml   # configs escolhidas
sbatch scripts/jobs/stats.sbatch                        # estatísticas (só CPU)
```

Detalhes do ambiente do CIn (partição, GPUs, VPN) em [docs/SERVIDOR.md](docs/SERVIDOR.md).

### 6. Tabelas e figuras

```bash
python scripts/aggregate_results.py   # results/tabela_{reproducao,custo}.{csv,tex}
python scripts/make_figures.py        # results/figuras/*.pdf e deteccao_por_ataque.{csv,tex}
python scripts/tsne_figures.py        # t-SNE (precisa do model.ckpt e do cache dos dados)
```

As tabelas são gravadas também em LaTeX, prontas para o relatório.

## Configuração dos experimentos

Cada arquivo em [`configs/`](configs/) é um YAML plano com os hiperparâmetros da
Tabela 14 do artigo. As chaves ausentes usam os padrões de
[`graphids/config.py`](graphids/config.py).

| Chave | Significado |
|---|---|
| `dataset`, `data_dir`, `fraction`, `drop_timestamps`, `seed` | dados e amostragem |
| `edim_out`, `nhops`, `fanout`, `agg_type`, `gnn_dropout` | E-GraphSAGE |
| `num_layers`, `num_heads`, `embed_dim`, `ff_dim`, `window_size`, `mask_ratio`, `ae_dropout` | Transformer |
| `learning_rate`, `gnn_weight_decay`, `ae_weight_decay`, `gnn_batch_size`, `ae_batch_size` | otimização |
| `num_epochs`, `patience`, `grad_clip`, `threshold_steps` | treino, early stopping e threshold |
| `device`, `output_dir` | execução |

A config do NF-ToN-IoT-v3 parte da do NF-UNSW-NB15-v3 (mesmas features NetFlow v3),
com `fraction: 0.2`.

## Testes e qualidade de código

```bash
pytest -q
ruff check .
ruff format --check graphids tests
```

O CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) roda lint e testes em CPU
a cada push na `main` e em cada pull request.

## Documentação complementar

- [Plano de desenvolvimento](docs/PLANO.md): escopo, metas de reprodução, decisões e cronograma
- [Contratos entre módulos](docs/CONTRATOS.md): formato dos dados, dos resultados, da config e das métricas
- [Servidor do CIn](docs/SERVIDOR.md): ambiente, datasets e rodadas no cluster
- [Notas sobre o código oficial](docs/REFERENCIA-CODIGO-OFICIAL.md): o que aprendemos lendo a implementação dos autores
- [Como contribuir](docs/CONTRIBUINDO.md)

A implementação é nossa. O [código oficial](https://github.com/lorenzo9uerra/GraphIDS)
(Apache 2.0) serviu só como referência: nenhum arquivo dele foi copiado.

## Equipe

Equipe 10 da disciplina CIN0114 (2026.2), Prof. Paulo Freitas: Rafael (coordenação),
Danilo, Bernardo e Felipe. A divisão do trabalho está em [docs/frentes/](docs/frentes/).

## Referências

1. L. Guerra, T. Chapuis, G. Duc, P. Mozharovskyi e V.-T. Nguyen, "Self-Supervised
   Learning of Graph Representations for Network Intrusion Detection", em *Advances in
   Neural Information Processing Systems (NeurIPS)*, vol. 38, 2025. Código oficial:
   <https://github.com/lorenzo9uerra/GraphIDS>.
2. Datasets NetFlow para NIDS da Universidade de Queensland:
   <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>.
