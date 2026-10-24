"""Testes do pipeline de dados da F2 com um CSV NetFlow falso e pequeno.

O CSV imita o formato dos datasets v3 da UQ (IPs, portas, features numéricas,
timestamps, Label e Attack), para testar o pipeline sem os arquivos reais.
"""

import numpy as np
import pandas as pd
import pytest

from graphids.data.loading import (
    ATTACK_COL,
    IP_COLS,
    LABEL_COL,
    TIMESTAMP_COLS,
    csv_path,
    load_flow_table,
)
from graphids.data.split import (
    sample_and_split,
    split_counts,
    split_indices,
    stratified_sample,
)

DATASET = "NF-FAKE-v3"
# Quantidade de fluxos por categoria. Todas têm fluxos suficientes para
# aparecer em treino, validação e teste.
CLASS_SIZES = {"Benign": 800, "DoS": 100, "Exploits": 60, "Worms": 40}


def make_fake_csv(data_dir, seed=0, with_timestamps=True):
    """Grava ``<data_dir>/NF-FAKE-v3/NF-FAKE-v3.csv`` e devolve o DataFrame."""
    rng = np.random.default_rng(seed)
    attack = np.concatenate([[name] * n for name, n in CLASS_SIZES.items()])
    rng.shuffle(attack)
    n = len(attack)
    df = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": [f"10.0.0.{i}" for i in rng.integers(0, 20, n)],
            "L4_SRC_PORT": rng.integers(1024, 65535, n),
            "IPV4_DST_ADDR": [f"10.0.1.{i}" for i in rng.integers(0, 5, n)],
            "L4_DST_PORT": rng.choice([22, 53, 80, 443], n),
            "PROTOCOL": rng.choice([6, 17], n),
            "IN_BYTES": rng.exponential(1000, n),
            "OUT_BYTES": rng.exponential(500, n),
        }
    )
    if with_timestamps:
        df["FLOW_START_MILLISECONDS"] = 1_700_000_000_000 + np.arange(n)
        df["FLOW_END_MILLISECONDS"] = df["FLOW_START_MILLISECONDS"] + 10
    # Valores inválidos, como acontece nos CSVs reais.
    df.loc[0, "IN_BYTES"] = np.inf
    df.loc[1, "OUT_BYTES"] = -np.inf
    df.loc[2, "IN_BYTES"] = np.nan
    df[LABEL_COL] = (attack != "Benign").astype(int)
    df[ATTACK_COL] = attack

    folder = data_dir / DATASET
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{DATASET}.csv", index=False)
    return df


@pytest.fixture
def table(tmp_path):
    make_fake_csv(tmp_path)
    return load_flow_table(tmp_path, DATASET, drop_timestamps=True)


# --- 2.1 Leitura e limpeza ---------------------------------------------------


def test_csv_ausente_da_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="NF-INEXISTENTE"):
        csv_path(tmp_path, "NF-INEXISTENTE")


def test_features_sem_ips_rotulos_e_timestamps(table):
    proibidas = {*IP_COLS, LABEL_COL, ATTACK_COL, *TIMESTAMP_COLS}
    assert not proibidas & set(table.feature_names)
    assert table.feature_names == [
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "PROTOCOL",
        "IN_BYTES",
        "OUT_BYTES",
    ]
    # IPs, Label e Attack continuam na tabela, fora das features.
    for col in (*IP_COLS, LABEL_COL, ATTACK_COL):
        assert col in table.flows.columns


def test_timestamps_mantidos_quando_pedido(tmp_path):
    make_fake_csv(tmp_path)
    table = load_flow_table(tmp_path, DATASET, drop_timestamps=False)
    assert set(TIMESTAMP_COLS) <= set(table.feature_names)


def test_inf_e_nan_viram_zero(table):
    x = table.features()
    assert np.isfinite(x).all()
    assert x[0, table.feature_names.index("IN_BYTES")] == 0
    assert x[1, table.feature_names.index("OUT_BYTES")] == 0
    assert x[2, table.feature_names.index("IN_BYTES")] == 0


def test_tipos_economizam_memoria(table):
    assert table.features().dtype == np.float32
    for col in table.feature_names:
        assert table.flows[col].dtype == np.float32
    for col in (*IP_COLS, ATTACK_COL):
        assert isinstance(table.flows[col].dtype, pd.CategoricalDtype)


# --- 2.2 Amostragem e split --------------------------------------------------


def test_split_reprodutivel(table):
    a = split_indices(table, seed=3)
    b = split_indices(table, seed=3)
    c = split_indices(table, seed=4)
    for part in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(a, part), getattr(b, part))
    assert not np.array_equal(a.val, c.val)


def test_treino_sem_ataques(table):
    splits = split_indices(table, seed=1)
    labels = table.flows[LABEL_COL].to_numpy()
    assert (labels[splits.train] == 0).all()
    # Validação e teste têm ataques.
    assert labels[splits.val].sum() > 0
    assert labels[splits.test].sum() > 0


def test_splits_disjuntos_e_proporcoes(table):
    splits = split_indices(table, seed=1)
    tr, va, te = set(splits.train), set(splits.val), set(splits.test)
    assert not (tr & va or tr & te or va & te)
    n = table.num_flows
    n_benign = CLASS_SIZES["Benign"]
    # 80% dos benignos no treino, 10% do total em validação e em teste.
    assert len(splits.train) == pytest.approx(0.8 * n_benign, abs=2)
    assert len(splits.val) == pytest.approx(0.1 * n, abs=2)
    assert len(splits.test) == pytest.approx(0.1 * n, abs=2)


def test_split_estratificado_por_ataque(table):
    splits = split_indices(table, seed=1)
    counts = split_counts(table, splits)
    for name, total in CLASS_SIZES.items():
        # Cada categoria tem ~10% dos seus fluxos na validação e no teste.
        assert counts.loc[name, "val"] == pytest.approx(0.1 * total, abs=1)
        assert counts.loc[name, "test"] == pytest.approx(0.1 * total, abs=1)
    assert (counts.drop(index="Benign")["train"] == 0).all()


def test_amostra_estratificada(table):
    sampled = stratified_sample(table, fraction=0.5, seed=1)
    counts = sampled.flows[ATTACK_COL].value_counts()
    for name, total in CLASS_SIZES.items():
        assert counts[name] == pytest.approx(0.5 * total, abs=1)
    # Índices renumerados e features preservadas.
    assert sampled.flows.index.tolist() == list(range(sampled.num_flows))
    assert sampled.feature_names == table.feature_names


def test_amostra_desligada_devolve_tabela_inteira(table):
    assert stratified_sample(table, fraction=None, seed=1) is table
    assert stratified_sample(table, fraction=1.0, seed=1) is table


def test_amostra_acontece_antes_do_split(table):
    sampled, splits = sample_and_split(table, seed=1, fraction=0.5)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    n_benignos_treino = CLASS_SIZES["Benign"] // 2 * 0.8
    # Os índices se referem à amostra, não à tabela original.
    assert max(splits.val.max(), splits.test.max()) < sampled.num_flows
    assert total < sampled.num_flows  # os ataques do treino foram removidos
    assert len(splits.train) == pytest.approx(n_benignos_treino, abs=2)
