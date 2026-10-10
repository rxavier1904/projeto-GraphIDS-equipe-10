# F3: Infraestrutura no servidor do CIn e execução dos experimentos

**Responsável:** Bernardo · **Peso:** ~20% · **Módulos:** `scripts/`, `results/`, `requirements*.txt` / ambiente

Garante que o código rode no servidor do CIn, executa todas as rodadas de reprodução e
transforma os resultados nas tabelas do relatório.

## Entregas

### 3.1 Ambiente (semana 1)
- Ambiente Python no servidor (venv, conda ou uv) com PyTorch, PyG e `pyg_lib` na versão
  CUDA do servidor.
- Documentar em `docs/SERVIDOR.md`: como acessar, carregar o ambiente, onde ficam os
  dados e como submeter jobs.
- Script `scripts/check_env.py` que imprime a versão do CUDA, a GPU e as versões das
  bibliotecas.

### 3.2 Datasets (semana 1)
- `scripts/download_datasets.sh`: baixar e descompactar no servidor
  NF-UNSW-NB15-v2/v3, NF-CSE-CIC-IDS2018-v2/v3 (o arquivo atual se chama
  `NF-CICIDS2018-*`, então é preciso renomear) e NF-ToN-IoT-v3, todos de
  <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>.
- Conferir checksums e registrar o tamanho de cada arquivo.

### 3.3 Execução (semanas 2 e 3)
- Script de job para o gerenciador de filas do servidor (ex.: `scripts/jobs/train.sbatch`
  se for SLURM).
- `scripts/run_seeds.sh <config> <seeds...>`: roda várias seeds em sequência ou como
  array job.
- Primeira execução completa no NF-UNSW-NB15-v3 assim que a F1 liberar o laço de treino.
- **Rodadas oficiais**: 4 datasets × 3 seeds (por exemplo, 1, 2 e 3). Estimativa pela
  Tabela 2 do artigo: de 0,4 a 1,5 h por execução numa A100.
- Acompanhar falhas (memória, tempo) e reportar à F1.

### 3.4 Agregação: `scripts/aggregate_results.py` (semana 4)
- Ler `results/runs/**/metrics.json` (contrato C2).
- Gerar `results/tabela_reproducao.csv` e `.tex` com média ± desvio de PR-AUC e F1
  macro, lado a lado com a Tabela 3 do artigo, mostrando a diferença.
- Tabela de custo: parâmetros, tempo de treino, inferência em µs por amostra e pico de
  memória (comparar com as Tabelas 2 e 4).

### 3.5 Relatório
Seção 6 (configuração experimental: hardware, software, seeds e hiperparâmetros) e
apoio à seção 7 com as tabelas de reprodução.

## Testes
- `aggregate_results.py` testado com `metrics.json` sintéticos.

## Depende de
- F1 para a CLI e o laço de treino. F2 para o pipeline de dados.
