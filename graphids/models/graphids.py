"""Modelo GraphIDS completo: E-GraphSAGE + Transformer MAE (Figura 3 do artigo).

Fluxo de um mini-batch de B fluxos:

1. E-GraphSAGE gera h_1..h_B (um embedding por fluxo, com contexto de 1 hop);
2. os embeddings são cortados em janelas de ``window_size`` fluxos;
3. as janelas passam pelo Transformer em lotes de ``ae_batch_size`` janelas;
4. o score de anomalia de cada fluxo é o erro quadrático s_i = ||h_i - ĥ_i||² (Eq. 1).

O treino minimiza a média de s_i (Eq. 2) e o gradiente atravessa os dois
componentes, então o GNN aprende embeddings úteis para a reconstrução.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from graphids.models.egraphsage import EGraphSAGE
from graphids.models.transformer_mae import TransformerMAE


def to_windows(h: torch.Tensor, window_size: int):
    """Corta [B, d] em janelas [W, window_size, d], completando com zeros.

    Returns:
        windows: [W, window_size, d].
        padding: [W, window_size] bool, ``True`` nas posições completadas.
    """
    b, d = h.shape
    n_windows = (b + window_size - 1) // window_size
    total = n_windows * window_size
    if total != b:
        h = torch.cat([h, h.new_zeros(total - b, d)])
    padding = torch.arange(total, device=h.device) >= b
    return h.view(n_windows, window_size, d), padding.view(n_windows, window_size)


@dataclass
class Reconstruction:
    """Saída de ``GraphIDS.reconstruct``."""

    loss: torch.Tensor  # escalar: média do erro quadrático por elemento (Eq. 2)
    scores: torch.Tensor  # [B]: s_i = ||h_i - ĥ_i||² de cada fluxo (Eq. 1)


class GraphIDS(nn.Module):
    """GraphIDS: detector de intrusão auto-supervisionado.

    Args:
        in_features: número de features de cada fluxo.
        edim_out: dimensão do embedding do GNN.
        embed_dim, num_heads, num_layers, ff_dim: hiperparâmetros do Transformer.
        gnn_dropout, ae_dropout: dropout de cada componente.
        window_size: fluxos por janela do Transformer.
        mask_ratio: proporção de ligações de atenção desligadas no treino.
        detach_target: se ``True``, o alvo da reconstrução não propaga gradiente.
            O código oficial deixa o gradiente passar (padrão ``False``).
    """

    def __init__(
        self,
        in_features: int,
        edim_out: int = 64,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        ff_dim: int = 256,
        gnn_dropout: float = 0.0,
        ae_dropout: float = 0.0,
        window_size: int = 512,
        mask_ratio: float = 0.15,
        detach_target: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.detach_target = detach_target
        self.encoder = EGraphSAGE(in_features, edim_out, gnn_dropout)
        self.autoencoder = TransformerMAE(
            edim_out, embed_dim, num_heads, num_layers, ff_dim, ae_dropout, mask_ratio
        )

    @classmethod
    def from_config(cls, cfg, in_features: int) -> GraphIDS:
        return cls(
            in_features=in_features,
            edim_out=cfg.edim_out,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            ff_dim=cfg.ff_dim,
            gnn_dropout=cfg.gnn_dropout,
            ae_dropout=cfg.ae_dropout,
            window_size=cfg.window_size,
            mask_ratio=cfg.mask_ratio,
            detach_target=cfg.detach_target,
        )

    def embed(self, node_feats, src_pos, dst_pos) -> torch.Tensor:
        """Embeddings dos fluxos do batch: [B, edim_out]."""
        return self.encoder(node_feats, src_pos, dst_pos)

    def reconstruct(self, h: torch.Tensor, ae_batch_size: int = 64) -> Reconstruction:
        """Reconstrói os embeddings ``h`` [B, d] janela a janela.

        A perda de cada lote de janelas é a média do erro quadrático sobre os
        elementos válidos (sem padding). A perda final é a média entre os lotes,
        como no código oficial, e gera um único ``backward`` por batch do GNN.
        """
        windows, padding = to_windows(h, self.window_size)
        target = windows.detach() if self.detach_target else windows
        chunk_losses, chunk_scores = [], []
        for start in range(0, windows.shape[0], ae_batch_size):
            w = windows[start : start + ae_batch_size]
            pad = padding[start : start + ae_batch_size]
            recon = self.autoencoder(w, pad)
            sq = (recon - target[start : start + ae_batch_size]) ** 2  # [W, L, d]
            valid = (~pad).unsqueeze(-1)
            chunk_losses.append((sq * valid).sum() / (valid.sum() * sq.shape[-1]))
            chunk_scores.append(sq.sum(-1)[~pad])  # Eq. (1), só posições válidas
        return Reconstruction(
            loss=torch.stack(chunk_losses).mean(), scores=torch.cat(chunk_scores)
        )

    def forward(self, node_feats, src_pos, dst_pos, ae_batch_size: int = 64):
        """Atalho: embeddings + reconstrução de um batch de fluxos."""
        return self.reconstruct(self.embed(node_feats, src_pos, dst_pos), ae_batch_size)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
