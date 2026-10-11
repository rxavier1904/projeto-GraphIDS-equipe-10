"""Autoencoder Transformer com máscara de atenção (Seção 3.3 do artigo).

Recebe janelas de embeddings de fluxos (até 512 por janela) e tenta
reconstruí-las. Detalhes que seguem o artigo:

- os embeddings são projetados para uma dimensão menor (``embed_dim``);
- sem positional encoding, porque os fluxos são embaralhados (Apêndice C.2);
- o encoder gera representações contextuais e o decoder usa a MESMA sequência
  projetada como consulta, com cross-attention na saída do encoder;
- no treino, uma máscara binária simétrica desliga uma fração ``mask_ratio`` das
  ligações de atenção. Nenhum token é removido: é um dropout estruturado na
  atenção. Na inferência a máscara é desligada.
"""

from __future__ import annotations

import torch
from torch import nn


def random_attention_mask(
    size: int, ratio: float, device=None, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Máscara simétrica [size, size]; ``True`` = ligação desligada.

    Sorteia o triângulo superior (sem a diagonal) e espelha. A diagonal nunca é
    mascarada, então toda posição consegue olhar pelo menos para si mesma e a
    softmax nunca fica sem entradas.
    """
    upper = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), 1)
    draw = torch.rand(size, size, device=device, generator=generator) < ratio
    mask = upper & draw
    return mask | mask.T


class TransformerMAE(nn.Module):
    """Encoder-decoder Transformer que reconstrói janelas de embeddings.

    Args:
        input_dim: dimensão dos embeddings do GNN (``edim_out``).
        embed_dim: dimensão interna do Transformer (``d_model``).
        num_heads, num_layers, ff_dim, dropout: hiperparâmetros dos blocos.
        mask_ratio: proporção de ligações de atenção desligadas no treino.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        ff_dim: int = 256,
        dropout: float = 0.0,
        mask_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.input_projection = nn.Linear(input_dim, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, ff_dim, dropout, batch_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            embed_dim, num_heads, ff_dim, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.output_projection = nn.Linear(embed_dim, input_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.input_projection, self.output_projection):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for name, param in list(self.encoder.named_parameters()) + list(
            self.decoder.named_parameters()
        ):
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, windows: torch.Tensor, padding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reconstrói um lote de janelas.

        Args:
            windows: [W, L, input_dim], W janelas de até L fluxos.
            padding: [W, L] bool, ``True`` nas posições de preenchimento.

        Returns:
            [W, L, input_dim], reconstrução de cada embedding.
        """
        x = self.input_projection(windows)
        attn_mask = None
        if self.training and self.mask_ratio > 0:
            attn_mask = random_attention_mask(x.shape[1], self.mask_ratio, x.device)

        memory = self.encoder(x, mask=attn_mask, src_key_padding_mask=padding)
        out = self.decoder(
            x,
            memory,
            tgt_mask=attn_mask,
            tgt_key_padding_mask=padding,
            memory_key_padding_mask=padding,
        )
        return self.output_projection(out)
