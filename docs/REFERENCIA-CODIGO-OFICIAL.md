# Notas sobre o código oficial (referência, não copiar)

Repositório: <https://github.com/lorenzo9uerra/GraphIDS>, branch `main` (PyG), commit
`2072b7c`. Licença Apache 2.0. Existe também a branch `dgl`, usada para os números
exatos do artigo. Estas notas resumem **o que o código faz**, para que a nossa
implementação do zero siga o mesmo comportamento. Os detalhes que o artigo não deixa
claros estão marcados com ⚠.

## Pré-processamento (`utils/dataloaders.py`): F2

1. Lê `<data_dir>/<dataset>/<dataset>.csv`.
2. Se `fraction` estiver definido (0,2 no CIC-IDS2018), faz uma amostra estratificada
   por `Attack` **antes** do split.
3. Separa `Attack`/`Label` das features, troca ±inf por NaN e NaN por 0.
4. Features das arestas: todas as colunas menos `IPV4_SRC_ADDR` e `IPV4_DST_ADDR`. Nos v3,
   também tira `FLOW_START_MILLISECONDS` e `FLOW_END_MILLISECONDS`.
   ⚠ Portas (`L4_SRC_PORT`, `L4_DST_PORT`) e campos categóricos como `PROTOCOL` e
   `L7_PROTO` entram como números contínuos.
5. Split estratificado por `Attack`: 80% de treino e 20% do resto, que depois vira
   10%/10% de validação e teste (dois `train_test_split` com `random_state=seed`).
6. Remove os ataques do treino.
7. `MinMaxScaler` ajustado nas features do treino benigno. Validação e teste passam por
   `clip(-10, 10)`.
8. ⚠ **Mapa de IPs compartilhado**: os índices de nó são criados com a união dos IPs
   dos três splits. Cada split vira um grafo separado (as arestas não se misturam),
   mas os nós têm o mesmo índice.
9. Feature dos nós: vetor de 1s (não é usada de fato, ver abaixo).

## Modelo (`models/graphids.py`): F1

**E-GraphSAGE (`SAGELayer`), 1 camada**
- Mensagem = feature da aresta. Agregação = **média das arestas que chegam** em cada nó
  de destino (`scatter mean`).
- `h_nó = ReLU(W_neigh · média)`, com `W_neigh: F → F`.
  ⚠ A feature do próprio nó não é concatenada, diferente do GraphSAGE clássico.
- Embedding da aresta (u→v): `h_e = Dropout(W_edge · [h_u ‖ h_v])`, com
  `W_edge: 2F → edim_out`.
  ⚠ A aresta só aparece no próprio embedding de forma indireta, pela agregação em v.
- Inicialização Xavier normal com ganho de ReLU.

**Transformer Autoencoder**
- `input_projection: edim_out → embed_dim`, sem positional encoding no padrão.
- `nn.TransformerEncoder` e `nn.TransformerDecoder` do PyTorch, `num_layers=1`,
  `nhead=4`, `dim_feedforward=256`, `batch_first=True`.
- O decoder recebe **a mesma sequência projetada** como `tgt` e a saída do encoder como
  `memory`.
- **Máscara de atenção** (só no treino): sorteia uma matriz triangular superior com
  probabilidade `mask_ratio` e simetriza (`mask | mask.T`). A mesma máscara vai como
  `mask` do encoder e como `tgt_mask` do decoder. O `memory_mask` não é usado.
- Máscara de padding: ignora as posições preenchidas na última janela.
- `output_projection: embed_dim → edim_out`. Inicialização Xavier uniforme, viés zero.

## Treino (`utils/trainers.py`): F1

- `LinkNeighborLoader` do PyG sobre as arestas do grafo de treino, `batch_size`
  16.384 ou 32.768, `num_neighbors=[fanout]` (32.768), com shuffle. `drop_last=True` no
  treino **e na validação** ⚠.
- Para cada lote do GNN: calcula os embeddings, corta em janelas de 512, monta
  sub-lotes de 64 janelas, soma a MSE (com máscara de padding) dos sub-lotes, tira a
  média e faz **um único `backward` por lote do GNN**.
  ⚠ O gradiente passa **também pelo alvo** (não há stop-gradient). O comentário diz
  que assim convergia mais rápido.
- `clip_grad_norm_(1.0)`, AdamW com dois grupos: GNN com weight decay 0,6 e
  Transformer com 0,01 a 0,05.
- Validação a cada época: PR-AUC (`average_precision_score`) sobre o erro de
  reconstrução. O checkpoint é salvo quando `val_pr_auc >= melhor`. Early stopping
  com paciência 20.
- **Threshold**: varre 500 valores em `linspace(min, max)` dos erros de validação e
  fica com o maior F1 macro. Existe também um modo não supervisionado
  (mediana + 10·MAD) que não é usado nos resultados.
- **Score de anomalia** de cada fluxo: média do erro quadrático entre as dimensões
  (`calculate_errors`), que é proporcional a ‖h − ĥ‖².
- Seeds fixas para `random`, `numpy` e `torch`. Seed padrão 24.

## Hiperparâmetros (Tabela 14 do artigo e `configs/*.yaml`)

| Parâmetro | UNSW-v3 | CIC-v3 | UNSW-v2 | CIC-v2 |
|---|---|---|---|---|
| edim_out | 96 | 64 | 72 | 64 |
| gnn dropout | 0,6 | 0,5 | 0,75 | 0,5 |
| embed_dim | 48 | 32 | 48 | 32 |
| ae_dropout | 0,0 | 0,2 | 0,0 | 0,2 |
| learning_rate | 1e-4 | 1e-4 | 1,1e-5 | 7,4e-5 |
| gnn_weight_decay | 0,6 | 0,6 | 0,6 | 0,6 |
| ae_weight_decay | 0,04 | 0,04 | 0,046 | 0,011 |
| gnn_batch_size | 16.384 | 16.384 | 32.768 | 16.384 |
| fraction | — | 0,2 | — | 0,2 |

Comuns aos quatro: nhops 1, fanout 32.768, agregação mean, 1 camada, 4 cabeças, janela
512, mask_ratio 0,15, ae_batch_size 64, 100 épocas no máximo.

## Pontos para discutir no relatório

- Como o embedding do fluxo u→v depende só das médias de entrada de u e de v, dois
  fluxos entre o mesmo par de hosts no mesmo batch recebem **o mesmo embedding**.
  O fluxo só se distingue pelo efeito que causa na vizinhança do destino.
- O alvo da reconstrução é o próprio embedding do GNN, que também está sendo treinado.
  Isso cria risco de colapso (embeddings triviais com erro zero). O weight decay alto
  no GNN e o dropout parecem segurar isso.
- O threshold e o early stopping usam rótulos da validação. Os autores reconhecem essa
  limitação.
- O ataque de Infiltration no CIC-IDS2018 é quase todo perdido (taxa de detecção de
  0,18 a 0,25).
