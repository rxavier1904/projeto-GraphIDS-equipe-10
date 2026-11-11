"""Geração de figuras e extração de embeddings para o relatório do GraphIDS.

Este módulo implementa as visualizações descritas no artigo (Guerra et al.,
NeurIPS 2025) e na especificação da F4:

- Figura 4: Curva Precisão-Revocação (individual e comparativa entre datasets).
- Figuras 5 a 8: Distribuição do score de anomalia por categoria de ataque em
  escala logarítmica com o limiar ótimo (:math:`\\tau`) e gráfico de taxas de
  detecção por ataque.
- Figuras 9 a 12: Extração de embeddings do E-GraphSAGE a partir do checkpoint
  e projeção t-SNE por categoria de ataque.

Todos os scores e metadados são lidos dos artefatos do contrato C2
(``scores_*.npz``, ``metrics.json`` e ``model.ckpt``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve

from graphids.config import load_config
from graphids.data import load_splits
from graphids.evaluation.metrics import detection_rate_per_attack
from graphids.models.graphids import GraphIDS
from graphids.models.sampling import NeighborhoodSampler
from graphids.training.trainer import resolve_device


def load_scores(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Carrega os arrays gravados em ``scores_val.npz`` ou ``scores_test.npz``.

    Args:
        npz_path: Caminho para o arquivo ``.npz``.

    Returns:
        Dicionário com ``errors``, ``labels``, ``attacks`` e ``attack_names``.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "errors": data["errors"],
            "labels": data["labels"],
            "attacks": data["attacks"],
            "attack_names": data["attack_names"],
        }


def _load_threshold(
    metrics_path: str | Path | None, threshold: float | None
) -> float | None:
    """Resolve o limiar a partir do argumento ou de ``metrics.json``."""
    if threshold is not None:
        return float(threshold)
    if metrics_path is not None:
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        if "threshold" in metrics:
            return float(metrics["threshold"])
    return None


def plot_precision_recall_curve(
    scores_path: str | Path,
    output_path: str | Path | None = None,
    dataset_name: str | None = None,
) -> plt.Figure:
    """Gera a curva Precisão-Revocação de um dataset (Figura 4 do artigo).

    Args:
        scores_path: Caminho para o arquivo ``scores_test.npz``.
        output_path: Se fornecido, salva a imagem em PNG/PDF.
        dataset_name: Nome do dataset para exibir no título e legenda.

    Returns:
        O objeto ``Figure`` do Matplotlib.
    """
    data = load_scores(scores_path)
    labels = data["labels"]
    scores = data["errors"]

    precision, recall, _ = precision_recall_curve(labels, scores)
    auc_val = average_precision_score(labels, scores)
    prevalence = float(np.mean(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    name = dataset_name or Path(scores_path).parent.parent.name
    ax.plot(
        recall,
        precision,
        color="#1f77b4",
        lw=2,
        label=f"{name} (PR-AUC = {auc_val:.4f})",
    )
    # Linha de base teórica (prevalência da classe positiva)
    ax.axhline(
        y=prevalence,
        color="gray",
        linestyle="--",
        label=f"Aleatório (Prevalência = {prevalence:.3f})",
    )

    ax.set_xlabel("Revocação (Recall)", fontsize=11)
    ax.set_ylabel("Precisão (Precision)", fontsize=11)
    ax.set_title(f"Curva Precisão-Revocação - {name}", fontsize=12, pad=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    return fig


def plot_pr_curves_comparison(
    runs: Mapping[str, str | Path],
    output_path: str | Path | None = None,
    title: str = "Curvas Precisão-Revocação por Dataset",
) -> plt.Figure:
    """Compara curvas Precisão-Revocação de múltiplos datasets num único gráfico.

    Útil para reproduzir a Figura 4 do artigo, contrastando os datasets de
    referência com o ``NF-ToN-IoT-v3``.

    Args:
        runs: Mapeamento ``{nome_dataset: caminho_para_scores_test.npz}``.
        output_path: Se fornecido, salva a imagem em PNG/PDF.
        title: Título do gráfico.

    Returns:
        O objeto ``Figure`` do Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, path in runs.items():
        data = load_scores(path)
        labels = data["labels"]
        scores = data["errors"]
        precision, recall, _ = precision_recall_curve(labels, scores)
        auc_val = average_precision_score(labels, scores)
        ax.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {auc_val:.4f})")

    ax.set_xlabel("Revocação (Recall)", fontsize=11)
    ax.set_ylabel("Precisão (Precision)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    return fig


def plot_anomaly_score_distribution(
    scores_path: str | Path,
    metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
    threshold: float | None = None,
    dataset_name: str | None = None,
) -> plt.Figure:
    """Gera a distribuição de erro de anomalia por categoria (Figuras 5 a 8 do artigo).

    Plota os scores contínuos de erro de reconstrução em escala logarítmica para cada
    categoria de ataque (e tráfego benigno), acompanhado da linha tracejada que marca
    o limiar ótimo de decisão (:math:`\\tau`).

    Args:
        scores_path: Caminho para ``scores_test.npz``.
        metrics_path: Caminho para ``metrics.json`` (lê o limiar se omitido).
        output_path: Caminho de destino para salvar a figura.
        threshold: Valor escalar do limiar :math:`\\tau` (opcional).
        dataset_name: Nome do dataset para o título.

    Returns:
        O objeto ``Figure`` do Matplotlib.
    """
    data = load_scores(scores_path)
    errors = data["errors"]
    attacks = data["attacks"]
    attack_names = list(data["attack_names"])
    tau = _load_threshold(metrics_path, threshold)

    rates = (
        detection_rate_per_attack(errors, attacks, tau, attack_names)
        if tau is not None
        else {}
    )

    name = dataset_name or Path(scores_path).parent.parent.name
    fig, ax = plt.subplots(figsize=(max(9, len(attack_names) * 1.05), 6))

    # Coloca Benign primeiro e ordena os ataques por mediana decrescente.
    present_indices = [
        idx for idx in range(len(attack_names)) if np.any(attacks == idx)
    ]
    benign_idx = [i for i in present_indices if attack_names[i].lower() == "benign"]
    attack_idx = sorted(
        [i for i in present_indices if attack_names[i].lower() != "benign"],
        key=lambda i: float(np.median(errors[attacks == i])),
        reverse=True,
    )
    ordered = benign_idx + attack_idx

    data_by_attack = []
    labels_present = []
    is_benign = []
    for idx in ordered:
        atk_name = attack_names[idx]
        subset = errors[attacks == idx]
        # Clip mínimo positivo para visualização estável em escala log
        data_by_attack.append(np.clip(subset, 1e-7, None))
        benign_flag = atk_name.lower() == "benign"
        is_benign.append(benign_flag)

        if tau is not None:
            if benign_flag:
                fpr = float(np.mean(subset > tau)) * 100
                labels_present.append(f"{atk_name}\n(N={len(subset)}, FPR={fpr:.1f}%)")
            else:
                dr = rates.get(atk_name, 0.0) * 100
                labels_present.append(f"{atk_name}\n(N={len(subset)}, DR={dr:.1f}%)")
        else:
            labels_present.append(f"{atk_name}\n(N={len(subset)})")

    bp = ax.boxplot(
        data_by_attack,
        tick_labels=labels_present,
        patch_artist=True,
        showfliers=False,
    )

    for patch, benign_flag in zip(bp["boxes"], is_benign, strict=True):
        if benign_flag:
            patch.set_facecolor("#a1d99b")
            patch.set_edgecolor("#2ca02c")
        else:
            patch.set_facecolor("#aec7e8")
            patch.set_edgecolor("#1f77b4")

    ax.set_yscale("log")
    ax.set_ylabel("Score de Anomalia (Erro MSE em Escala Log)", fontsize=11)
    ax.set_title(f"Distribuição do Score por Categoria - {name}", fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if tau is not None and tau > 0:
        ax.axhline(
            y=tau,
            color="#d62728",
            linestyle="--",
            lw=1.5,
            label=f"Limiar de Decisão ($\\tau$ = {tau:.4f})",
        )
        ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    return fig


def plot_detection_rates(
    scores_path: str | Path,
    metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
    threshold: float | None = None,
    dataset_name: str | None = None,
) -> plt.Figure:
    """Plota gráfico de barras horizontais da taxa de detecção por ataque.

    Args:
        scores_path: Caminho para ``scores_test.npz``.
        metrics_path: Caminho para ``metrics.json`` (lê o limiar se omitido).
        output_path: Caminho para gravação da figura.
        threshold: Valor escalar do limiar :math:`\\tau`.
        dataset_name: Nome do dataset para o título.

    Returns:
        O objeto ``Figure`` do Matplotlib.
    """
    data = load_scores(scores_path)
    tau = _load_threshold(metrics_path, threshold)
    if tau is None:
        raise ValueError("Informe `threshold` ou `metrics_path` com a chave threshold.")

    rates = detection_rate_per_attack(
        data["errors"], data["attacks"], tau, list(data["attack_names"])
    )
    sorted_items = sorted(rates.items(), key=lambda x: x[1])
    names = [k for k, _ in sorted_items]
    values = [v * 100.0 for _, v in sorted_items]

    name = dataset_name or Path(scores_path).parent.parent.name
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5 + 1.5)))
    bars = ax.barh(names, values, color="#1f77b4", alpha=0.85)
    ax.set_xlim([0.0, 108.0])
    ax.set_xlabel("Taxa de Detecção (%)", fontsize=11)
    ax.set_title(f"Taxa de Detecção por Ataque - {name}", fontsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            val + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    return fig


def stratified_sample_indices(
    attacks: np.ndarray, max_samples: int, seed: int = 42
) -> np.ndarray:
    """Seleciona até ``max_samples`` índices balanceando as categorias presentes.

    Garante que ataques raros (como ``ransomware`` e ``mitm``) mantenham uma
    cota mínima visível no t-SNE em vez de serem engolidos pelas classes
    majoritárias.

    Args:
        attacks: Vetor inteiro [N] com a categoria de cada fluxo.
        max_samples: Orçamento máximo de amostras.
        seed: Semente para reprodutibilidade.

    Returns:
        Vetor ordenado de índices ``int64`` com tamanho ``<= max_samples``.
    """
    n_total = len(attacks)
    if n_total <= max_samples:
        return np.arange(n_total, dtype=np.int64)

    rng = np.random.default_rng(seed)
    classes = np.unique(attacks)
    per_class = max(1, max_samples // len(classes))

    selected: list[np.ndarray] = []
    leftover_pool: list[np.ndarray] = []
    for cls in classes:
        cls_idx = np.flatnonzero(attacks == cls)
        if len(cls_idx) <= per_class:
            selected.append(cls_idx)
        else:
            perm = rng.permutation(cls_idx)
            selected.append(perm[:per_class])
            leftover_pool.append(perm[per_class:])

    chosen = np.concatenate(selected)
    remaining = max_samples - len(chosen)
    if remaining > 0 and leftover_pool:
        pool = np.concatenate(leftover_pool)
        extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
        chosen = np.concatenate([chosen, extra])

    return np.sort(chosen).astype(np.int64)


@torch.inference_mode()
def extract_embeddings(
    config_path: str | Path,
    checkpoint_path: str | Path,
    split: str = "test",
    max_samples: int = 5000,
    seed: int = 1,
    data_dir: str | Path | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extrai embeddings do E-GraphSAGE a partir de um checkpoint treinado.

    Carrega o grafo do split escolhido (aproveitando o cache da F2), amostra
    fluxos estratificados por categoria de ataque e executa apenas
    ``model.embed`` em modo de inferência.

    Args:
        config_path: YAML do experimento (pasta ``configs/``).
        checkpoint_path: Caminho para o ``model.ckpt`` salvo no treino.
        split: Split a amostrar (``"train"``, ``"val"`` ou ``"test"``).
        max_samples: Número máximo de fluxos amostrados para o t-SNE.
        seed: Semente (usada para localizar o split/cache e amostrar).
        data_dir: Sobrescreve ``data_dir`` da config, se fornecido.
        device: Dispositivo PyTorch (``"auto"``, ``"cpu"`` ou ``"cuda"``).

    Returns:
        Tupla ``(embeddings [M, edim_out], attacks [M], attack_names)``.
    """
    overrides = [f"seed={seed}"]
    if data_dir is not None:
        overrides.append(f"data_dir={data_dir}")
    cfg = load_config(config_path, overrides)

    torch_device = resolve_device(device)
    splits = load_splits(cfg.data_config())
    graph = getattr(splits, split)

    model = GraphIDS.from_config(cfg, splits.num_features).to(torch_device)
    state = torch.load(checkpoint_path, map_location=torch_device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    attacks_all = graph.edge_attack.cpu().numpy()
    sample_idx = stratified_sample_indices(attacks_all, max_samples, seed=seed)

    graph_dev = graph.to(torch_device)
    sampler = NeighborhoodSampler(graph_dev, cfg.fanout)
    generator = torch.Generator().manual_seed(seed)

    idx_tensor = torch.from_numpy(sample_idx)
    chunks: list[np.ndarray] = []
    for start in range(0, len(idx_tensor), cfg.gnn_batch_size):
        batch_ids = idx_tensor[start : start + cfg.gnn_batch_size].to(torch_device)
        node_feats, src_pos, dst_pos = sampler.batch(batch_ids, generator)
        emb = model.embed(node_feats, src_pos, dst_pos)
        chunks.append(emb.cpu().numpy())

    embeddings = (
        np.concatenate(chunks, axis=0)
        if chunks
        else np.empty((0, cfg.edim_out), dtype=np.float32)
    )
    return embeddings, attacks_all[sample_idx], list(graph.attack_names)


def plot_tsne_projection(
    embeddings: np.ndarray,
    attacks: np.ndarray,
    attack_names: Sequence[str],
    output_path: str | Path | None = None,
    max_samples: int = 3000,
    random_state: int = 42,
    dataset_name: str | None = None,
) -> plt.Figure:
    """Gera projeção bidimensional t-SNE dos embeddings (Figuras 9 a 12 do artigo).

    Args:
        embeddings: Matriz de representações latentes [N, D] dos fluxos.
        attacks: Índices das categorias [N].
        attack_names: Nomes legíveis das categorias.
        output_path: Caminho para gravação da figura.
        max_samples: Limite de amostras para viabilizar execução rápida do t-SNE.
        random_state: Semente do gerador pseudoaleatório.
        dataset_name: Identificação do dataset.

    Returns:
        O objeto ``Figure`` gerado.
    """
    n_samples = len(embeddings)
    if n_samples > max_samples:
        indices = stratified_sample_indices(
            np.asarray(attacks), max_samples, seed=random_state
        )
        X_sub = embeddings[indices]
        y_sub = np.asarray(attacks)[indices]
    else:
        X_sub = embeddings
        y_sub = np.asarray(attacks)

    perplexity = min(30, max(2, len(X_sub) - 1))
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    coords = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(9, 7))
    unique_atks = np.unique(y_sub)
    cmap = plt.get_cmap("tab10" if len(unique_atks) <= 10 else "tab20")

    for i, atk_idx in enumerate(unique_atks):
        mask = y_sub == atk_idx
        name = (
            attack_names[atk_idx] if atk_idx < len(attack_names) else f"Atk-{atk_idx}"
        )
        color = "#2ca02c" if name.lower() == "benign" else cmap(i % cmap.N)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=color,
            label=f"{name} (n={int(np.sum(mask))})",
            alpha=0.65,
            s=18,
            edgecolors="none",
        )

    title = "Projeção t-SNE dos Embeddings de Fluxo"
    if dataset_name:
        title += f" ({dataset_name})"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE Dimensão 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimensão 2", fontsize=10)
    ax.legend(loc="best", markerscale=1.5, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    return fig
