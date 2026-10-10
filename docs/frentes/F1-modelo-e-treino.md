# F1: Núcleo do modelo, treino e integração

**Responsável:** Rafael · **Peso:** ~40% · **Módulos:** `graphids/models/`, `graphids/training/`, `graphids/cli.py`, `configs/` (base)

É a frente central do projeto. Implementa do zero o sistema proposto no artigo
(seção 3), o treino ponta a ponta (seção 4.3), define os contratos que as outras
frentes usam e integra tudo. Também coordena a equipe e o relatório.

## Entregas

### 1.1 Esqueleto e contratos (semana 1)
- Pacote `graphids/` com `pyproject.toml` (ou `requirements.txt`), `ruff`, `pytest` e CI
  no GitHub Actions rodando os testes sintéticos.
- Dataclasses `DataConfig`, `ModelConfig`, `TrainConfig` e carregamento do YAML (contrato C3).
- Stub de `load_splits` que gera um grafo sintético, para todos poderem testar antes
  da F2 ficar pronta.

### 1.2 E-GraphSAGE: `graphids/models/egraphsage.py` (semana 1)
- Camada de passagem de mensagens (`torch_geometric.nn.MessagePassing`): a mensagem é
  a feature da aresta, agregada por **média** nos nós de destino.
- `h_v = ReLU(W_neigh · mean_{e→v} x_e)`.
- Embedding da aresta u→v: `h_e = Dropout(W_edge · [h_u ‖ h_v])`.
- Suporte a `nhops ≥ 1` (o artigo usa 1; mais hops ficam opcionais).
- Inicialização Xavier com ganho de ReLU.

### 1.3 Transformer Masked Autoencoder: `graphids/models/transformer_mae.py` (semana 2)
- Projeção linear `edim_out → embed_dim`, encoder e decoder com `num_layers` blocos,
  4 cabeças e FFN de 256.
- O decoder usa a sequência projetada como *query* e faz cross-attention na saída do
  encoder (seção 3.3).
- **Máscara de atenção simétrica** com proporção `mask_ratio` (0,15), aplicada só no
  treino. É dropout estruturado na atenção, sem remover tokens.
- Máscara de padding para a última janela incompleta.
- Projeção de volta para `edim_out`.

### 1.4 Modelo completo: `graphids/models/graphids.py` (semana 2)
- Junta 1.2 e 1.3. O `forward` recebe um subgrafo amostrado e devolve os embeddings
  `h`, as reconstruções `ĥ` e os scores `s_i = ‖h_i − ĥ_i‖²` (Eq. 1).
- Contagem de parâmetros, para comparar com a Tabela 2 (~0,57 M a 0,87 M).

### 1.5 Treino e inferência: `graphids/training/` (semanas 2 e 3)
- `LinkNeighborLoader` com `fanout`, embaralhamento e `gnn_batch_size`.
- Corte dos embeddings em janelas de 512 e lotes de 64 janelas.
  Perda MSE (Eq. 2) com máscara de padding e **um backward por lote do GNN**, passando
  pelos dois componentes.
- AdamW com dois grupos de weight decay (GNN e Transformer) e `clip_grad_norm(1.0)`.
- Validação por época com PR-AUC, checkpoint do melhor modelo e early stopping
  (paciência 20).
- **Threshold**: maximiza o F1 macro na validação e aplica no teste.
- Medição do tempo de treino, do tempo de inferência por amostra (µs) e do pico de
  memória da GPU.
- Seeds fixas e gravação dos artefatos do contrato C2.

### 1.6 CLI e configs (semana 2)
- `python -m graphids.cli --config configs/<dataset>.yaml --seed <s> [--test]`.
- Um YAML por dataset do artigo com os hiperparâmetros da Tabela 14.

### 1.7 Integração e reprodução (semanas 3 e 4)
- Rodar ponta a ponta com o pipeline real da F2 e depurar até o NF-UNSW-NB15-v3 chegar
  perto de 0,9998 de PR-AUC.
- Investigar diferenças em relação à Tabela 3 junto com o Bernardo e documentar as
  hipóteses.
- Revisar os PRs das outras frentes.

### 1.8 Relatório
Resumo, introdução, trabalhos relacionados, modelo de ameaça, sistema do artigo
(com figuras e equações), discussão dos resultados nos dados do artigo, conclusão e
revisão final.

## Testes
- Formas dos tensores em cada componente.
- A máscara de atenção é simétrica, tem a proporção esperada e fica desligada em `eval()`.
- O gradiente chega aos parâmetros do GNN depois de um passo.
- *Overfit* num grafo sintético minúsculo: a perda cai e os ataques sintéticos ficam com
  score maior.

## Depende de
- F2 para os dados reais (até lá, usa o stub sintético).
- F4 para as métricas por ataque (o treino usa sklearn direto).
