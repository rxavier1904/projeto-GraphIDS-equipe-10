"""Treino ponta a ponta e inferência do GraphIDS (Seção 4.3 do artigo).

Cada época:
1. embaralha os fluxos de treino (só benignos) e os divide em batches do GNN;
2. para cada batch: amostra a vizinhança, gera os embeddings, reconstrói as
   janelas e faz UM backward que atualiza o GNN e o Transformer juntos;
3. avalia na validação (PR-AUC), salva o melhor checkpoint e aplica early stopping.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from graphids.data.types import FlowGraph, GraphSplits
from graphids.models.graphids import GraphIDS
from graphids.models.sampling import NeighborhoodSampler
from graphids.training.threshold import best_f1_threshold


def set_seed(seed: int) -> None:
    """Fixa as sementes de Python, NumPy e PyTorch (reprodutibilidade)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_optimizer(model: GraphIDS, cfg) -> torch.optim.Optimizer:
    """AdamW com weight decay separado para o GNN e para o Transformer.

    O artigo observa que o GNN precisa de regularização bem mais forte
    (weight decay até 0.6) que o autoencoder (0.01 a 0.05).
    """
    return torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "weight_decay": cfg.gnn_weight_decay,
            },
            {
                "params": model.autoencoder.parameters(),
                "weight_decay": cfg.ae_weight_decay,
            },
        ],
        lr=cfg.learning_rate,
    )


def edge_batches(
    num_edges: int,
    batch_size: int,
    generator: torch.Generator,
    drop_last: bool = False,
) -> Iterator[torch.Tensor]:
    """Índices de fluxos embaralhados, em batches de ``batch_size``.

    O embaralhamento faz cada janela do Transformer conter fluxos de partes
    diferentes do grafo (Seção 3.3).
    """
    perm = torch.randperm(num_edges, generator=generator)
    for start in range(0, num_edges, batch_size):
        ids = perm[start : start + batch_size]
        if drop_last and len(ids) < batch_size and start > 0:
            break
        yield ids


@dataclass
class EvalResult:
    """Scores de um split, na ordem original dos fluxos."""

    scores: np.ndarray  # [E] score de anomalia de cada fluxo
    labels: np.ndarray  # [E] 0 = benigno, 1 = ataque
    attacks: np.ndarray  # [E] índice da categoria de ataque
    loss: float  # MSE médio (Eq. 2)
    seconds: float  # tempo total da passagem

    @property
    def pr_auc(self) -> float:
        """PR-AUC = average precision do scikit-learn (Seção 4.5)."""
        return float(average_precision_score(self.labels, self.scores))


def train_epoch(
    model: GraphIDS,
    sampler: NeighborhoodSampler,
    optimizer: torch.optim.Optimizer,
    cfg,
    generator: torch.Generator,
) -> float:
    """Uma época de treino. Devolve a perda média dos batches."""
    model.train()
    device = sampler.graph.edge_attr.device
    losses = []
    for ids in edge_batches(
        sampler.graph.num_edges, cfg.gnn_batch_size, generator, drop_last=True
    ):
        node_feats, src_pos, dst_pos = sampler.batch(ids.to(device), generator)
        out = model(node_feats, src_pos, dst_pos, cfg.ae_batch_size)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()  # gradiente atravessa Transformer E GNN (Seção 3.4)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        losses.append(out.loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.inference_mode()
def evaluate(
    model: GraphIDS, sampler: NeighborhoodSampler, cfg, seed: int = 0
) -> EvalResult:
    """Calcula o score de todos os fluxos de um split (sem máscara de atenção).

    A ordem dos fluxos também é embaralhada aqui, como no código oficial, mas com
    semente fixa para que a avaliação seja determinística entre épocas.
    """
    model.eval()
    graph = sampler.graph
    device = graph.edge_attr.device
    generator = torch.Generator().manual_seed(seed)
    scores = torch.empty(graph.num_edges, device=device)
    losses = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for ids in edge_batches(graph.num_edges, cfg.gnn_batch_size, generator):
        ids = ids.to(device)
        out = model(*sampler.batch(ids, generator), cfg.ae_batch_size)
        scores[ids] = out.scores
        losses.append(out.loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return EvalResult(
        scores=scores.cpu().numpy(),
        labels=graph.edge_label.cpu().numpy(),
        attacks=graph.edge_attack.cpu().numpy(),
        loss=float(np.mean(losses)),
        seconds=seconds,
    )


@dataclass
class FitResult:
    best_epoch: int
    best_val_pr_auc: float
    threshold: float
    train_time_s: float
    history: list[dict] = field(default_factory=list)


def save_checkpoint(path, model, optimizer, epoch, threshold, val_pr_auc) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "threshold": threshold,
            "val_pr_auc": val_pr_auc,
        },
        path,
    )


def fit(
    model: GraphIDS,
    splits: GraphSplits,
    cfg,
    checkpoint_path,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> FitResult:
    """Treina com early stopping pela PR-AUC de validação (paciência ``patience``).

    O checkpoint é sobrescrito sempre que a PR-AUC de validação empata ou melhora,
    junto com o threshold escolhido naquela época. Ao final, o melhor checkpoint é
    recarregado em ``model``.
    """
    train_graph: FlowGraph = splits.train.to(device)
    val_graph: FlowGraph = splits.val.to(device)
    train_sampler = NeighborhoodSampler(train_graph, cfg.fanout)
    val_sampler = NeighborhoodSampler(val_graph, cfg.fanout)
    optimizer = build_optimizer(model, cfg)
    generator = torch.Generator().manual_seed(cfg.seed)

    best_pr_auc, best_epoch, best_threshold, wait = -1.0, 0, float("nan"), 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_epoch(model, train_sampler, optimizer, cfg, generator)
        val = evaluate(model, val_sampler, cfg, seed=cfg.seed)
        val_pr_auc = val.pr_auc
        threshold, val_f1 = best_f1_threshold(
            val.scores, val.labels, cfg.threshold_steps
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val.loss,
                "val_pr_auc": val_pr_auc,
                "val_f1_macro": val_f1,
            }
        )
        log(
            f"época {epoch:3d} | perda treino {train_loss:.5f} | perda val "
            f"{val.loss:.5f} | PR-AUC val {val_pr_auc:.4f} | F1 val {val_f1:.4f}"
        )

        if val_pr_auc >= best_pr_auc:
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, threshold, val_pr_auc
            )
        if val_pr_auc > best_pr_auc:
            best_pr_auc, best_epoch, best_threshold, wait = (
                val_pr_auc,
                epoch,
                threshold,
                0,
            )
        else:
            if val_pr_auc == best_pr_auc:  # empate: checkpoint mais novo foi salvo
                best_epoch, best_threshold = epoch, threshold
            wait += 1
            if wait >= cfg.patience:
                log(f"early stopping na época {epoch}")
                break
    train_time = time.perf_counter() - start

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return FitResult(best_epoch, best_pr_auc, best_threshold, train_time, history)
