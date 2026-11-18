"""Figuras t-SNE dos embeddings de cada dataset (Figuras 9 a 12 do artigo).

Para cada ``results/runs/<dataset>/seed<s>/`` com ``model.ckpt``:

1. recarrega a config da execução, o split de teste (do cache em data/cache/)
   e o melhor checkpoint;
2. amostra até ``--per-class`` fluxos de teste de cada categoria;
3. calcula os embeddings do E-GraphSAGE e a saída do encoder do Transformer e
   projeta com t-SNE, junto com as features brutas dos mesmos fluxos (ver
   graphids/evaluation/embeddings.py para a leitura de cada painel);
4. grava ``results/figuras/<dataset>_tsne.pdf`` e os dados em
   ``results/figuras/<dataset>_tsne.npz`` (para refazer a figura sem GPU).

Uso:
    python scripts/tsne_figures.py                         # todos, seed 1
    python scripts/tsne_figures.py --dataset NF-ToN-IoT-v3 --seed 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from graphids.config import load_config
from graphids.data import load_splits
from graphids.evaluation.embeddings import (
    Projection,
    flow_embeddings,
    plot_projections,
    stratified_sample,
    transformer_latents,
    tsne_2d,
)
from graphids.models.graphids import GraphIDS
from graphids.training.trainer import resolve_device


def tsne_for_run(
    run: Path, out_dir: Path, per_class: int, data_dir: str | None
) -> Path:
    overrides = [f"data_dir={data_dir}"] if data_dir else []
    cfg = load_config(run / "config.yaml", overrides)
    device = resolve_device(cfg.device)
    splits = load_splits(cfg.data_config())
    test = splits.test

    model = GraphIDS.from_config(cfg, splits.num_features).to(device)
    state = torch.load(run / "model.ckpt", map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])

    attacks = test.edge_attack.cpu().numpy()
    ids = stratified_sample(attacks, per_class, seed=cfg.seed)
    print(f"{cfg.dataset} seed {cfg.seed}: {len(ids)} fluxos amostrados, {device}")

    emb = flow_embeddings(
        model, test.to(device), ids, cfg.fanout, cfg.gnn_batch_size, seed=cfg.seed
    )
    latent = transformer_latents(model, emb, cfg.window_size, seed=cfg.seed)
    raw = test.edge_attr[torch.as_tensor(ids)].cpu().numpy()
    xy_raw = tsne_2d(raw, seed=cfg.seed)
    xy_emb = tsne_2d(emb, seed=cfg.seed)
    xy_lat = tsne_2d(latent, seed=cfg.seed)
    n_pairs = len(np.unique(np.round(emb, 4), axis=0))
    print(f"  {n_pairs} embeddings distintos do GNN entre {len(ids)} fluxos")

    name = cfg.dataset
    np.savez_compressed(
        out_dir / f"{name}_tsne.npz",
        edge_ids=ids,
        attacks=attacks[ids],
        attack_names=np.array(test.attack_names),
        xy_embedding=xy_emb,
        xy_raw=xy_raw,
        xy_latent=xy_lat,
    )
    return plot_projections(
        [
            Projection("Features brutas (normalizadas)", xy_raw, attacks[ids]),
            Projection("Embeddings do E-GraphSAGE", xy_emb, attacks[ids]),
            Projection("Encoder do Transformer", xy_lat, attacks[ids]),
        ],
        list(test.attack_names),
        f"{name}: t-SNE do teste (seed {cfg.seed})",
        out_dir / f"{name}_tsne.pdf",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--out", default="results/figuras")
    parser.add_argument("--dataset", action="append", help="só estes (repete)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--per-class", type=int, default=1500)
    parser.add_argument("--data-dir", default=None, help="sobrescreve data_dir")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = args.dataset or sorted(
        p.name for p in Path(args.runs).iterdir() if p.is_dir()
    )
    failed = []
    for ds in datasets:
        run = Path(args.runs) / ds / f"seed{args.seed}"
        if not (run / "model.ckpt").exists():
            print(f"{ds}: sem {run / 'model.ckpt'}, pulando")
            continue
        try:
            print(f"  -> {tsne_for_run(run, out, args.per_class, args.data_dir)}")
        except Exception as exc:  # um dataset com problema não derruba os outros
            print(f"{ds}: ERRO {type(exc).__name__}: {exc}")
            failed.append(ds)
    if failed:
        raise SystemExit(f"falharam: {', '.join(failed)}")


if __name__ == "__main__":
    main()
