"""Execução completa de um experimento e gravação dos artefatos (contrato C2).

Cada execução grava em ``<output_dir>/<dataset>/seed<seed>/``:

- ``config.yaml``: configuração efetiva usada;
- ``model.ckpt``: melhor checkpoint (fica fora do git);
- ``history.json``: perdas e métricas de validação por época;
- ``scores_val.npz`` e ``scores_test.npz``: scores, rótulos e categorias;
- ``metrics.json``: métricas finais (lidas pela F3 para montar as tabelas).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from graphids.config import Config
from graphids.data import load_splits
from graphids.models.graphids import GraphIDS
from graphids.models.sampling import NeighborhoodSampler
from graphids.training.threshold import best_f1_threshold
from graphids.training.trainer import (
    EvalResult,
    evaluate,
    fit,
    resolve_device,
    set_seed,
)


def run_dir(cfg: Config) -> Path:
    return Path(cfg.output_dir) / cfg.dataset / f"seed{cfg.seed}"


def git_commit() -> str:
    """Commit atual do repositório, para rastrear qual código gerou o resultado."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def save_scores(path: Path, result: EvalResult, attack_names: list[str]) -> None:
    np.savez_compressed(
        path,
        errors=result.scores,
        labels=result.labels,
        attacks=result.attacks,
        attack_names=np.array(attack_names),
    )


def run_experiment(
    cfg: Config, test_only: bool = False, checkpoint: str | None = None, log=print
) -> dict:
    """Treina (ou só avalia) o GraphIDS e grava os artefatos da execução.

    Args:
        cfg: configuração do experimento.
        test_only: se ``True``, não treina; carrega ``checkpoint`` e avalia.
        checkpoint: caminho do checkpoint (padrão: ``<run_dir>/model.ckpt``).

    Returns:
        O conteúdo de ``metrics.json``.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(checkpoint) if checkpoint else out / "model.ckpt"
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)

    splits = load_splits(cfg.data_config())
    model = GraphIDS.from_config(cfg, splits.num_features).to(device)
    log(
        f"dataset {cfg.dataset} | {splits.num_features} features | "
        f"{model.num_parameters():,} parâmetros | dispositivo {device}"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if test_only:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        best_epoch, best_val, train_time = state["epoch"], state["val_pr_auc"], 0.0
        threshold = state["threshold"]
    else:
        result = fit(model, splits, cfg, ckpt_path, device, log)
        best_epoch, best_val = result.best_epoch, result.best_val_pr_auc
        threshold, train_time = result.threshold, result.train_time_s
        with open(out / "history.json", "w") as f:
            json.dump(result.history, f, indent=2)

    # Validação refeita com o melhor modelo, para salvar os scores e confirmar o
    # threshold (no modo só-teste, ele é recalculado a partir desses scores).
    val = evaluate(
        model,
        NeighborhoodSampler(splits.val.to(device), cfg.fanout),
        cfg,
        seed=cfg.seed,
    )
    if test_only or not np.isfinite(threshold):
        threshold, _ = best_f1_threshold(val.scores, val.labels, cfg.threshold_steps)

    test_graph = splits.test.to(device)
    test = evaluate(
        model, NeighborhoodSampler(test_graph, cfg.fanout), cfg, seed=cfg.seed
    )
    preds = (test.scores > threshold).astype(int)
    metrics = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "test_pr_auc": test.pr_auc,
        "test_f1_macro": float(
            f1_score(test.labels, preds, average="macro", zero_division=0)
        ),
        "val_best_pr_auc": float(best_val),
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "train_time_s": float(train_time),
        "inference_us_per_sample": test.seconds / max(test_graph.num_edges, 1) * 1e6,
        "peak_gpu_mem_mb": (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
        ),
        "num_params": model.num_parameters(),
        "git_commit": git_commit(),
    }
    save_scores(out / "scores_val.npz", val, splits.val.attack_names)
    save_scores(out / "scores_test.npz", test, splits.test.attack_names)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(
        f"TESTE | PR-AUC {metrics['test_pr_auc']:.4f} | "
        f"F1 macro {metrics['test_f1_macro']:.4f} | artefatos em {out}"
    )
    return metrics
