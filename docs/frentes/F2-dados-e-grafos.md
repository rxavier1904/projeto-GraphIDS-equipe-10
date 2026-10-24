# F2: Pipeline de dados e construção dos grafos

**Responsável:** Danilo · **Peso:** ~20% · **Módulos:** `graphids/data/`, `notebooks/estatisticas_datasets.ipynb`

Transforma os CSVs NetFlow brutos nos grafos de treino, validação e teste, seguindo a
seção 4.2 do artigo. Entrega o contrato **C1** ([CONTRATOS.md](../CONTRATOS.md)).

## Entregas

### 2.1 Leitura e limpeza: `graphids/data/loading.py` (semana 1)
- Ler `<data_dir>/<dataset>/<dataset>.csv` de forma eficiente (`dtype` explícito; o
  CIC-IDS2018 tem cerca de 20 M de linhas).
- Trocar ±inf por NaN e NaN por 0.
- Separar `Attack` e `Label` das features. Tirar os IPs das features. Nos v3, tirar
  também os timestamps (`FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS`).

### 2.2 Amostragem e split: `graphids/data/split.py` (semana 1)
- Amostra opcional estratificada por `Attack` (`fraction=0.2` no CIC-IDS2018), feita
  **antes** do split.
- Split estratificado por `Attack`: 80% de treino e 20% restantes divididos em 10%/10%
  de validação e teste, com `random_state=seed`.
- Remover os ataques do treino.

### 2.3 Normalização: `graphids/data/preprocessing.py` (semana 2)
- `MinMaxScaler` ajustado **só no treino benigno**, aplicado aos três splits.
- `clip(-10, 10)` na validação e no teste.
- Salvar o scaler junto do cache.

### 2.4 Construção do grafo: `graphids/data/graph.py` (semana 2)
- Nós = IPs, com mapa de índices **compartilhado** entre os três splits. Arestas
  dirigidas origem→destino, uma por fluxo.
- Um `FlowGraph` (`graphids/data/types.py`) por split, com `edge_index`, `edge_attr`,
  `edge_label`, `edge_attack` e `attack_names` (contrato C1).
- Implementar a leitura real dentro de `load_splits` (`graphids/data/loader.py`),
  hoje restrita ao dataset sintético.
- Cache em `<data_dir>/cache/<dataset>[_<fraction>]_seed<s>/`, para não reprocessar a
  cada execução.

### 2.5 Estatísticas: `graphids/data/stats.py` + notebook (semana 3)
A seção 6 do relatório exige estes números:
- número de fluxos, hosts e proporção de ataques (comparar com a Tabela 1 do artigo);
- **contagem por classe e por categoria de ataque em cada split** (treino, validação e
  teste);
- tabelas exportadas em CSV e LaTeX para o relatório.

### 2.6 Relatório
Seção 6 (Metodologia, dados do artigo): origem dos datasets, por que foram escolhidos,
o que representam, como os splits foram formados e as tabelas de contagem.

## Estado atual

Implementado e testado: 2.1 a 2.5. Arquivos em `graphids/data/`: `loading.py`,
`split.py`, `preprocessing.py`, `graph.py`, `cache.py`, `stats.py` e o
`load_splits` do `loader.py`, que agora aceita os datasets reais. Testes em
`tests/test_data_pipeline.py`. Conferido no NF-UNSW-NB15-v3: 1.790.185 fluxos de
treino, 236.542 de validação, 236.543 de teste, 44 hosts (igual à Tabela 1) e
49 features.

Falta a 2.6 (texto do relatório), que depende das tabelas de todos os datasets.

## Como usar

```bash
# grafos de um dataset (lê o CSV na 1ª vez e grava o cache em data/cache/)
python -c "from graphids.data import load_splits, DataConfig; \
s = load_splits(DataConfig('NF-UNSW-NB15-v3', data_dir='data', seed=1)); print(s.train)"

# tabelas da seção 6 (CSV + LaTeX em results/estatisticas/), uma ou mais configs
python -m graphids.data.stats --config configs/NF-UNSW-NB15-v3.yaml --seed 1
```

As configs definem `fraction` e `drop_timestamps`, então as estatísticas saem
exatamente com os dados que o modelo recebe. O cache fica em
`data/cache/<dataset>[_<fraction>]_seed<s>/`; para refazer, apague a pasta.

## Testes
- O split é estratificado e reprodutível (mesma seed, mesmos índices).
- O treino não tem nenhum ataque.
- O scaler só enxerga dados do treino. Os valores de validação e teste ficam em [−10, 10].
- No grafo, `edge_index` aponta para nós válidos e `num_nodes` é igual nos três splits.

## Depende de
- F3 para os CSVs no servidor (localmente, use um CSV pequeno ou sintético).
- F3 para gerar no servidor as estatísticas dos datasets grandes
  (NF-CSE-CIC-IDS2018 v2 e v3), que não cabem num notebook comum. Ver a
  entrega 3.2 da F3.
