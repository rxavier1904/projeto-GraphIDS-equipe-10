# Plano de desenvolvimento: GraphIDS (Equipe 10)

Disciplina **CIN0114: Técnicas de Ataque e Detecção de Intrusão (2026.2)**, Prof. Paulo Freitas.

Equipe: **Rafael** (coordenação), **Danilo**, **Bernardo** e **Felipe**.

## 1. Objetivo

Reproduzir o artigo **"Self-Supervised Learning of Graph Representations for Network
Intrusion Detection"** (Guerra et al., NeurIPS 2025), o GraphIDS. A implementação é
**nossa, feita do zero**. O código oficial
([lorenzo9uerra/GraphIDS](https://github.com/lorenzo9uerra/GraphIDS), Apache 2.0) serve
só como referência teórica e de contexto: não copiamos arquivos dele. O que aprendemos
lendo esse código está em [REFERENCIA-CODIGO-OFICIAL.md](REFERENCIA-CODIGO-OFICIAL.md).

## 2. Escopo e entregáveis (especificação da disciplina)

| Item | Status | Prazo |
|---|---|---|
| Reproduzir o artigo com resultados próximos aos publicados | obrigatório | 18/11/2026 |
| Rodar o GraphIDS em outro dataset (**NF-ToN-IoT-v3**), com justificativa | obrigatório | 18/11/2026 |
| Relatório em PDF (template Overleaf, formato artigo, refs IEEE) | obrigatório | 18/11/2026 |
| Link do GitHub com **todo o código comentado** | obrigatório | 18/11/2026 |
| Modificações para melhorar o sistema (ponto extra) + apresentação | **fora do escopo por ora** | 19/11/2026 |

### Metas de reprodução (Tabela 3 do artigo, média ± desvio entre seeds)

| Dataset | PR-AUC | F1 macro | Observação |
|---|---|---|---|
| NF-UNSW-NB15-v3 | 0,9998 ± 0,0007 | 0,9961 ± 0,0084 | dataset completo |
| NF-CSE-CIC-IDS2018-v3 | 0,8819 ± 0,0347 | 0,9447 ± 0,0213 | amostra estratificada de 20% |
| NF-UNSW-NB15-v2 | 0,8116 ± 0,0367 | 0,9264 ± 0,0217 | dataset completo |
| NF-CSE-CIC-IDS2018-v2 | 0,9201 ± 0,0238 | 0,9431 ± 0,0131 | amostra estratificada de 20% |

Critério de "próximo o suficiente" que vamos usar: média dentro de cerca de 2 desvios
do valor publicado, ou diferença justificada no relatório.

## 3. Decisões tomadas

- **Implementação do zero** em Python 3.11+, PyTorch e PyTorch Geometric (PyG) para o
  grafo e a amostragem de vizinhança.
- **Treinamento no servidor do CIn**, operado pelo Bernardo.
- **Novo dataset: NF-ToN-IoT-v3**. Vem do mesmo repositório da UQ dos datasets do artigo
  e tem o mesmo conjunto de 53 features NetFlow, mas cobre um ambiente IoT/IIoT com
  ataques diferentes.
- **Reprodução nos 4 datasets do artigo** (v2 e v3), com 3 seeds cada. Ainda falta
  confirmação da equipe.
- **Autoria**: cada integrante faz os commits da própria frente com a própria conta do
  GitHub. Nada de linhas automáticas de ferramentas (ver [CONTRIBUINDO.md](CONTRIBUINDO.md)).

## 4. As 4 frentes

| # | Frente | Responsável | Peso | Documento |
|---|---|---|---|---|
| F1 | **Núcleo do modelo, treino e integração** | **Rafael** | ~40% | [F1-modelo-e-treino.md](frentes/F1-modelo-e-treino.md) |
| F2 | Pipeline de dados e construção dos grafos | Danilo | ~20% | [F2-dados-e-grafos.md](frentes/F2-dados-e-grafos.md) |
| F3 | Infraestrutura no servidor do CIn e execução dos experimentos | Bernardo | ~20% | [F3-infra-e-experimentos.md](frentes/F3-infra-e-experimentos.md) |
| F4 | Novo dataset (NF-ToN-IoT-v3), avaliação e figuras | Felipe | ~20% | [F4-novo-dataset-e-avaliacao.md](frentes/F4-novo-dataset-e-avaliacao.md) |

A F1 é a frente central. Ela contém o sistema proposto pelo artigo (E-GraphSAGE,
autoencoder Transformer com máscara de atenção e treino ponta a ponta), define os
contratos que as outras frentes usam, integra tudo e coordena o relatório.

As frentes conversam por contratos fixos, descritos em [CONTRATOS.md](CONTRATOS.md).
Assim todo mundo trabalha em paralelo desde a semana 1, usando dados sintéticos até o
pipeline real ficar pronto.

## 5. Estrutura do repositório

```
graphids/
  data/          F2  leitura, limpeza, split, normalização, grafo, estatísticas
  models/        F1  E-GraphSAGE, Transformer MAE, GraphIDS
  training/      F1  laço de treino, early stopping, threshold, checkpoint
  evaluation/    F4  métricas por ataque, curvas PR, distribuição de scores, t-SNE
  cli.py         F1  ponto de entrada: python -m graphids.cli --config ...
configs/         F1 (base), F4 (ToN-IoT)  um YAML por dataset
scripts/         F3  download, jobs do servidor, várias seeds, agregação
notebooks/       F2 (estatísticas), F4 (EDA e figuras)
results/         F3  métricas por execução (JSON) e tabelas agregadas
tests/           todos  cada frente testa o próprio módulo com dados sintéticos
docs/            plano, frentes, contratos, notas de reprodução
report/          PDF final e link do Overleaf
```

## 6. Cronograma

| Semana | Até | F1 Rafael | F2 Danilo | F3 Bernardo | F4 Felipe |
|---|---|---|---|---|---|
| 1 | 17/10 | Esqueleto do pacote, contratos, E-GraphSAGE | Leitura e limpeza dos CSVs, split estratificado | Ambiente no servidor, download dos 5 datasets | EDA do NF-ToN-IoT-v3, justificativa da escolha |
| 2 | 24/10 | Transformer MAE, laço de treino, threshold, CLI | Normalização, construção do grafo, cache, amostra de 20% | Scripts de job e de seeds, 1ª execução no UNSW-v3 | Métricas por ataque e figuras (com dados sintéticos) |
| 3 | 31/10 | Integração ponta a ponta, depuração até bater o UNSW-v3 | Estatísticas por classe e split (seção 6 do relatório) | Rodadas completas: 4 datasets × 3 seeds | Config e ajuste leve de hiperparâmetros no ToN-IoT |
| 4 | 07/11 | Análise das diferenças em relação ao artigo, revisão de código | Testes e documentação do pipeline | Agregação das tabelas, tempos e memória | Rodadas no ToN-IoT, figuras finais |
| 5 | 14/11 | Relatório: introdução, sistema, discussão, revisão final | Relatório: metodologia (dados) | Relatório: configuração experimental | Relatório: resultados no novo dataset |
| — | 18/11 | **Entrega** do relatório + GitHub | | | |

## 7. Divisão do relatório

| Seção | Responsável principal |
|---|---|
| Resumo, 1 Introdução, 2 Trabalhos relacionados | Rafael |
| 3 Modelo de ameaça | Rafael, com apoio do Felipe (ataques do ToN-IoT) |
| 4 Sistema do artigo de referência | Rafael |
| 6 Metodologia: dados do artigo (splits, contagem por classe) | Danilo |
| 6 Metodologia: novo dataset e justificativa | Felipe |
| 6 Metodologia: ambiente, seeds e hiperparâmetros | Bernardo |
| 7 Resultados: dados do artigo | Rafael e Bernardo |
| 7 Resultados: novo dataset | Felipe |
| 8 Conclusão, 9 Referências, revisão final | Rafael |

## 8. Perguntas em aberto

1. Reproduzir os 4 datasets (v2 e v3) ou só os v3? Por padrão, vamos fazer os 4.
2. Logins do GitHub do Danilo, Bernardo e Felipe, para dar acesso de escrita e atribuir as issues.
3. Gerenciador de filas do servidor do CIn (SLURM ou outro), para o Bernardo adaptar os scripts.
