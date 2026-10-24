"""Amostragem e divisão em treino, validação e teste.

Segue a Seção 4.2 do artigo e o código oficial:

1. amostra opcional estratificada por ``Attack`` (``fraction``), feita ANTES
   do split;
2. split estratificado por ``Attack``: 80% de treino e os 20% restantes
   divididos ao meio entre validação e teste (dois ``train_test_split`` com
   ``random_state=seed``);
3. remoção dos ataques do treino, que fica só com fluxos benignos.

Estratificar por ``Attack`` (e não só por ``Label``) garante que cada
categoria de ataque apareça na validação e no teste na mesma proporção do
dataset original.

As funções devolvem índices de linha, e não cópias da tabela: no
NF-CSE-CIC-IDS2018 cada cópia custaria alguns GB de memória.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from graphids.data.loading import ATTACK_COL, LABEL_COL, FlowTable


@dataclass
class SplitIndices:
    """Posições das linhas de ``FlowTable.flows`` em cada split.

    Atributos (arrays ``int64`` ordenados de forma crescente):
        train: só fluxos benignos.
        val: benignos e ataques.
        test: benignos e ataques.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_sample(table: FlowTable, fraction: float | None, seed: int) -> FlowTable:
    """Sorteia ``fraction`` dos fluxos, mantendo a proporção de cada ataque.

    Args:
        table: fluxos lidos por ``loading.read_netflow_csv``.
        fraction: proporção a manter, entre 0 e 1. ``None`` ou 1 devolve a
            tabela inteira.
        seed: semente do sorteio.

    Returns:
        Nova ``FlowTable`` com os índices renumerados de 0 a E'-1 e na ordem
        original do CSV.

    Observação: categorias com pouquíssimos fluxos podem sumir da amostra
    (por exemplo, 2 fluxos × 0,2 arredonda para 0). Nos datasets do projeto
    isso não acontece.
    """
    if fraction is None or fraction >= 1:
        return table
    if not 0 < fraction < 1:
        raise ValueError(f"fraction deve estar entre 0 e 1, recebido {fraction}")

    flows = table.flows
    sampled = flows.groupby(ATTACK_COL, observed=True, group_keys=False).sample(
        frac=fraction, random_state=seed
    )
    # sort_index devolve a ordem original do CSV; reset_index renumera.
    sampled = sampled.sort_index().reset_index(drop=True)
    # Remove do tipo category os IPs e ataques que não sobraram na amostra.
    for col in sampled.select_dtypes("category").columns:
        sampled[col] = sampled[col].cat.remove_unused_categories()
    return FlowTable(flows=sampled, feature_names=list(table.feature_names))


def _stratified_split(
    indices: np.ndarray, strata: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``train_test_split`` estratificado, com mensagem de erro mais clara."""
    try:
        return train_test_split(
            indices, test_size=test_size, stratify=strata, random_state=seed
        )
    except ValueError as err:
        names, counts = np.unique(strata, return_counts=True)
        rare = {str(n): int(c) for n, c in zip(names, counts, strict=True) if c < 2}
        raise ValueError(
            "Não foi possível estratificar por Attack: cada categoria precisa "
            f"de pelo menos 2 fluxos em cada etapa do split. Categorias com "
            f"poucos fluxos: {rare or 'nenhuma abaixo de 2'}. Erro original: {err}"
        ) from err


def split_indices(table: FlowTable, seed: int) -> SplitIndices:
    """Divide os fluxos em treino (80%), validação (10%) e teste (10%).

    Args:
        table: fluxos (já amostrados, se for o caso).
        seed: ``random_state`` dos dois ``train_test_split``. A mesma semente
            produz sempre os mesmos índices.

    Returns:
        ``SplitIndices`` com o treino já sem ataques.
    """
    attack = table.flows[ATTACK_COL].to_numpy()
    label = table.flows[LABEL_COL].to_numpy()
    all_idx = np.arange(table.num_flows)

    # 1º split: 80% treino, 20% para validação + teste.
    train_idx, rest_idx = _stratified_split(all_idx, attack, test_size=0.2, seed=seed)
    # 2º split: os 20% restantes ao meio, 10% validação e 10% teste.
    val_idx, test_idx = _stratified_split(
        rest_idx, attack[rest_idx], test_size=0.5, seed=seed
    )

    # O treino usa só tráfego benigno (Seção 4.2): o modelo aprende o normal.
    train_idx = train_idx[label[train_idx] == 0]

    return SplitIndices(
        train=np.sort(train_idx).astype(np.int64),
        val=np.sort(val_idx).astype(np.int64),
        test=np.sort(test_idx).astype(np.int64),
    )


def sample_and_split(
    table: FlowTable, seed: int, fraction: float | None = None
) -> tuple[FlowTable, SplitIndices]:
    """Aplica a amostra opcional e depois o split, nessa ordem.

    Returns:
        A tabela usada no split (a amostra, se ``fraction`` foi dado) e os
        índices de cada split, que se referem a essa tabela.
    """
    table = stratified_sample(table, fraction, seed)
    return table, split_indices(table, seed)


def split_counts(table: FlowTable, splits: SplitIndices) -> pd.DataFrame:
    """Contagem de fluxos por categoria de ataque em cada split.

    Útil para conferir o split agora e base da entrega 2.5 (estatísticas).
    Linhas = categorias de ataque; colunas = ``train``, ``val``, ``test``.
    """
    attack = table.flows[ATTACK_COL]
    counts = {
        name: attack.iloc[idx].value_counts()
        for name, idx in (
            ("train", splits.train),
            ("val", splits.val),
            ("test", splits.test),
        )
    }
    return pd.DataFrame(counts).fillna(0).astype(int)
