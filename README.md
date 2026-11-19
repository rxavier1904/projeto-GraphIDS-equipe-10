# GraphIDS: reproduction (Team 10, CIN0114 2026.2)

From-scratch reproduction of **"Self-Supervised Learning of Graph Representations for
Network Intrusion Detection"** (Guerra et al., NeurIPS 2025), plus a new dataset
(NF-ToN-IoT-v3) and an ablation of the E-GraphSAGE aggregation function.

GraphIDS is trained on benign traffic only. An E-GraphSAGE layer turns each NetFlow flow
into an embedding with its 1-hop host context, and a Transformer masked autoencoder
reconstructs those embeddings. Flows with a high reconstruction error are flagged as
attacks.

## Setup

Requires Python ≥ 3.10. A GPU is needed only for the real datasets.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # synthetic-data tests, a few seconds on CPU
```

## Data

The NetFlow datasets come from the University of Queensland
(<https://staff.itee.uq.edu.au/marius/NIDS_datasets/>). Put each zip in
`data/raw/<dataset>.zip` and unpack it with:

```bash
scripts/download_datasets.sh data                 # all 5 datasets
scripts/download_datasets.sh data NF-UNSW-NB15-v3 # just one
```

This produces `data/<dataset>/<dataset>.csv`. The first run of each dataset builds the
graphs and caches them in `data/cache/`.

## Running

```bash
# quick run on the synthetic graph
python -m graphids.cli --config configs/synthetic.yaml

# train + evaluate on a real dataset
python -m graphids.cli --config configs/NF-UNSW-NB15-v3.yaml --seed 1 --data-dir data

# override any hyperparameter
python -m graphids.cli --config configs/NF-UNSW-NB15-v3.yaml --set agg_type=sum

# evaluate an existing checkpoint only
python -m graphids.cli --config configs/NF-UNSW-NB15-v3.yaml --seed 1 --test

# several seeds in sequence (skips seeds that already finished)
scripts/run_seeds.sh configs/NF-UNSW-NB15-v3.yaml 1 2 3
```

Each run writes to `results/runs/<dataset>/seed<s>/`: `config.yaml`, `metrics.json`,
`history.json`, `scores_val.npz`, `scores_test.npz` and `model.ckpt`.

`agg_type` selects the host aggregation: `mean` (paper, default), `wmean`, `max`, `sum`
or `mean_max`.

### SLURM cluster

```bash
scripts/jobs/submit_all.sh       # 4 paper datasets × 3 seeds
scripts/jobs/submit_agg.sh       # aggregation ablation: 4 variants × 5 datasets, seed 1
```

### Tables and figures

```bash
python scripts/aggregate_results.py   # reproduction and cost tables (results/)
python scripts/make_figures.py        # PR curves, score histograms, per-attack detection
python scripts/tsne_figures.py        # t-SNE of the embeddings
python scripts/aggregate_agg.py       # aggregation ablation table
```

## Project structure

```
configs/            one YAML per dataset (hyperparameters from Table 14 of the paper)
graphids/
  cli.py            command-line entry point
  config.py         experiment configuration
  data/             CSV loading, split, Min-Max scaling, host graph and cache
  models/           E-GraphSAGE, neighbourhood sampler, Transformer MAE, GraphIDS
  training/         training loop, early stopping, threshold, full experiment run
  evaluation/       metrics, plots and embedding extraction
scripts/            datasets, seeds, tables, figures and environment check
  jobs/             SLURM job scripts
notebooks/          dataset statistics
tests/              unit and integration tests (synthetic data)
results/            runs, logs, tables and figures
```

## Team

Rafael Xavier (model and training) · Danilo Barrote (data and graphs) ·
Bernardo Siqueira (infrastructure and experiments) · Felipe Maran (NF-ToN-IoT-v3 and
evaluation)

## Reference

L. Guerra, T. Chapuis, G. Duc, P. Mozharovskyi and V.-T. Nguyen, "Self-Supervised
Learning of Graph Representations for Network Intrusion Detection", *NeurIPS*, vol. 38,
2025. Official code: <https://github.com/lorenzo9uerra/GraphIDS>.
