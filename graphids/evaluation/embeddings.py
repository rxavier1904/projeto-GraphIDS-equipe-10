"""t-SNE dos embeddings dos fluxos por categoria de ataque (Figuras 9 a 12).

A figura mostra, para os mesmos fluxos, três painéis:

1. features brutas normalizadas (``edge_attr``), como referência;
2. embeddings ``h_i`` do E-GraphSAGE (``GraphIDS.embed``), o que o Transformer
   tenta reconstruir. Pelo desenho do modelo (e do código oficial), ``h_i`` só
   depende da vizinhança dos dois hosts do fluxo, não das features do próprio
   fluxo: fluxos entre o mesmo par de hosts caem quase no mesmo ponto, então o
   painel mostra poucos aglomerados densos. Isso é um resultado, não um erro;
3. saída do encoder do Transformer, que mistura cada fluxo com os outros da
   janela e varia por fluxo.

O t-SNE é O(n²) em memória e tempo, então usamos uma amostra estratificada do
teste: até ``per_class`` fluxos de cada categoria (todas as categorias raras
entram inteiras).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from graphids.data.types import FlowGraph  # noqa: E402
from graphids.models.graphids import GraphIDS  # noqa: E402
from graphids.models.sampling import NeighborhoodSampler  # noqa: E402

# 20 cores distintas (tab20) cobrem as categorias de todos os datasets usados.
_CMAP = plt.get_cmap("tab20")


def stratified_sample(attacks: np.ndarray, per_class: int, seed: int = 0) -> np.ndarray:
    """Índices com até ``per_class`` fluxos de cada categoria (ordenados)."""
    rng = np.random.default_rng(seed)
    chosen = []
    for cat in np.unique(attacks):
        idx = np.flatnonzero(attacks == cat)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        chosen.append(idx)
    return np.sort(np.concatenate(chosen))


@torch.inference_mode()
def flow_embeddings(
    model: GraphIDS,
    graph: FlowGraph,
    edge_ids: np.ndarray,
    fanout: int,
    batch_size: int = 16384,
    seed: int = 0,
) -> np.ndarray:
    """Embeddings ``h_i`` [len(edge_ids), edim_out] dos fluxos pedidos.

    ``graph`` deve estar no mesmo dispositivo do modelo. A vizinhança de cada
    host é a do grafo inteiro do split (como na avaliação), não só a da amostra.
    """
    model.eval()
    device = graph.edge_attr.device
    sampler = NeighborhoodSampler(graph, fanout)
    generator = torch.Generator().manual_seed(seed)
    ids = torch.as_tensor(edge_ids, dtype=torch.long)
    out = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size].to(device)
        out.append(model.embed(*sampler.batch(batch, generator)).float().cpu())
    return torch.cat(out).numpy()


@torch.inference_mode()
def transformer_latents(
    model: GraphIDS, h: np.ndarray, window_size: int, seed: int = 0
) -> np.ndarray:
    """Saída do encoder do Transformer [n, embed_dim] para os embeddings ``h``.

    Os fluxos são embaralhados em janelas de ``window_size`` (como na avaliação)
    e a saída volta à ordem original. Observação: na avaliação real cada janela
    tem fluxos sorteados do split inteiro; aqui a janela só tem fluxos da
    amostra estratificada, que tem mais ataques que o teste.
    """
    from graphids.models.graphids import to_windows

    model.eval()
    device = next(model.parameters()).device
    x = torch.as_tensor(h, dtype=torch.float32, device=device)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    windows, padding = to_windows(x[perm.to(device)], window_size)
    ae = model.autoencoder
    memory = ae.encoder(ae.input_projection(windows), src_key_padding_mask=padding)
    latent = torch.empty(len(x), memory.shape[-1], device=device)
    latent[perm.to(device)] = memory[~padding]
    return latent.float().cpu().numpy()


def tsne_2d(x: np.ndarray, seed: int = 0, perplexity: float = 30.0) -> np.ndarray:
    """Projeção t-SNE 2D (inicialização PCA, determinística pela seed)."""
    perplexity = min(perplexity, max(5.0, (len(x) - 1) / 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(np.asarray(x, dtype=np.float32))


@dataclass
class Projection:
    """Coordenadas 2D de um conjunto de fluxos e a categoria de cada um."""

    title: str
    xy: np.ndarray  # [n, 2]
    attacks: np.ndarray  # [n]


def plot_projections(
    projections: list[Projection],
    attack_names: list[str],
    suptitle: str,
    path: str | Path,
) -> Path:
    """Um painel por projeção, cores por categoria; benignos ao fundo, em cinza."""
    fig, axes = plt.subplots(
        1,
        len(projections),
        figsize=(4.6 * len(projections), 4.4),
        layout="constrained",
        squeeze=False,
    )
    present = sorted(set(np.concatenate([p.attacks for p in projections]).tolist()))
    for ax, proj in zip(axes[0], projections, strict=True):
        for cat in present:
            mask = proj.attacks == cat
            if not mask.any():
                continue
            name = attack_names[cat]
            benign = name.lower() == "benign"
            ax.scatter(
                proj.xy[mask, 0],
                proj.xy[mask, 1],
                s=3 if benign else 4,
                alpha=0.35 if benign else 0.8,
                color="#b0b0b0" if benign else _CMAP(cat % 20),
                label=name,
                zorder=1 if benign else 2,
                rasterized=True,  # PDF leve mesmo com milhares de pontos
            )
        ax.set_title(proj.title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside right center",
        fontsize=8,
        markerscale=3,
        frameon=False,
    )
    fig.suptitle(suptitle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path
