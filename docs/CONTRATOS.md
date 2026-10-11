# Contratos entre as frentes

Estes são os pontos de encaixe entre os módulos. Se cada frente respeitar a interface
abaixo, pode trabalhar em paralelo usando dados sintéticos. Mudar um contrato exige
combinar com o Rafael (F1), que integra o sistema.

## C1: Dados → Modelo (F2 entrega, F1 consome)

```python
# graphids/data/__init__.py
def load_splits(cfg: DataConfig) -> GraphSplits: ...

@dataclass
class GraphSplits:
    train: torch_geometric.data.Data   # só fluxos benignos
    val:   torch_geometric.data.Data   # benignos + ataques (rotulado)
    test:  torch_geometric.data.Data   # benignos + ataques (rotulado)
    feature_names: list[str]           # ordem das colunas de edge_attr
    num_nodes: int                     # mapa de IPs compartilhado entre os splits
```

Cada `Data` contém:

| Campo | Forma | Descrição |
|---|---|---|
| `edge_index` | `[2, E]` long | origem e destino (índices de host) de cada fluxo |
| `edge_attr` | `[E, F]` float | features do fluxo normalizadas (Min-Max ajustado no treino benigno, clip em [−10, 10]) |
| `edge_label` | `[E]` long | 0 = benigno, 1 = ataque |
| `edge_attack` | `[E]` long | índice da categoria de ataque (para análise por ataque) |
| `attack_names` | `list[str]` | nome de cada índice de `edge_attack` |
| `num_nodes` | int | número de hosts |

`DataConfig` tem os campos `dataset`, `data_dir`, `fraction` (`None` ou 0,2), `seed` e
`drop_timestamps` (verdadeiro nos v3).

## C2: Execução → Resultados (F1 grava, F3 e F4 consomem)

Cada execução de `python -m graphids.cli --config configs/<dataset>.yaml --seed <s>` grava:

```
results/runs/<dataset>/seed<s>/
  metrics.json     # ver esquema abaixo
  scores_val.npz   # errors, labels, attacks  (validação)
  scores_test.npz  # errors, labels, attacks  (teste)
  config.yaml      # config efetiva usada
  model.ckpt       # melhor checkpoint (fica fora do git)
```

`metrics.json`:

```json
{
  "dataset": "NF-UNSW-NB15-v3",
  "seed": 1,
  "test_pr_auc": 0.0,
  "test_f1_macro": 0.0,
  "val_best_pr_auc": 0.0,
  "threshold": 0.0,
  "best_epoch": 0,
  "train_time_s": 0.0,
  "inference_us_per_sample": 0.0,
  "peak_gpu_mem_mb": 0.0,
  "num_params": 0,
  "git_commit": "abc1234"
}
```

## C3: Config (F1 define o esquema, F3 e F4 criam arquivos)

YAML plano com as chaves de hiperparâmetros da Tabela 14 do artigo: `edim_out`,
`nhops`, `fanout`, `agg_type`, `gnn_dropout`, `num_layers`, `num_heads`, `embed_dim`,
`window_size`, `mask_ratio`, `ae_dropout`, `learning_rate`, `gnn_weight_decay`,
`ae_weight_decay`, `gnn_batch_size`, `ae_batch_size`, `num_epochs`, `patience`, além das
chaves de dados do C1.

## C4: Métricas (F4 implementa, todas as frentes usam)

```python
# graphids/evaluation/metrics.py
pr_auc(labels, scores) -> float            # sklearn average_precision_score
f1_macro(labels, preds) -> float
detection_rate_per_attack(scores, attacks, threshold, attack_names) -> dict[str, float]
```
