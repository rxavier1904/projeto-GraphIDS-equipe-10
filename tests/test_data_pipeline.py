"""Testes do pipeline de dados da F2 com um CSV NetFlow falso e pequeno.

O CSV imita o formato dos datasets v3 da UQ (IPs, portas, features numéricas,
timestamps, Label e Attack), para testar o pipeline sem os arquivos reais.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from graphids.data.cache import cache_dir, load_cache
from graphids.data.graph import build_graphs
from graphids.data.loader import load_splits
from graphids.data.loading import (
    ATTACK_COL,
    IP_COLS,
    LABEL_COL,
    TIMESTAMP_COLS,
    csv_path,
    load_flow_table,
)
from graphids.data.preprocessing import fit_scaler, transform_features
from graphids.data.split import (
    sample_and_split,
    split_counts,
    split_indices,
    stratified_sample,
)
from graphids.data.types import DataConfig
from graphids.models.sampling import NeighborhoodSampler

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


# --- 2.3 Normalização --------------------------------------------------------


def test_scaler_usa_so_o_treino():
    train = np.array([[0.0, 5.0], [10.0, 5.0]], dtype=np.float32)
    val = np.array([[5.0, 5.0], [1000.0, 5.0], [-1000.0, 5.0]], dtype=np.float32)
    scaler = fit_scaler(train)
    np.testing.assert_allclose(scaler.data_min_, [0.0, 5.0])
    np.testing.assert_allclose(scaler.data_max_, [10.0, 5.0])
    out = transform_features(scaler, val)
    # 5 vira 0,5; valores extremos são cortados em ±10; coluna constante vira 0.
    np.testing.assert_allclose(out[:, 0], [0.5, 10.0, -10.0])
    np.testing.assert_allclose(out[:, 1], 0.0)
    assert out.dtype == np.float32


def test_normalizacao_nos_grafos(table):
    built = build_graphs(table, split_indices(table, seed=1))
    train = built.splits.train.edge_attr
    assert train.min() >= 0 and train.max() <= 1
    for g in (built.splits.val, built.splits.test):
        assert g.edge_attr.min() >= -10 and g.edge_attr.max() <= 10
    # O scaler foi ajustado com as linhas do treino, e só com elas.
    rows = split_indices(table, seed=1).train
    expected_min = table.flows[table.feature_names].to_numpy()[rows].min(0)
    np.testing.assert_allclose(built.scaler.data_min_, expected_min)


# --- 2.4 Grafo e cache -------------------------------------------------------


@pytest.fixture
def built(table):
    return build_graphs(table, split_indices(table, seed=1))


def test_grafos_seguem_o_contrato(built, table):
    splits = built.splits
    num_nodes = splits.train.num_nodes
    sizes = split_indices(table, seed=1)
    for name in ("train", "val", "test"):
        g = getattr(splits, name)
        assert g.num_nodes == num_nodes  # mapa de IPs compartilhado
        assert g.num_edges == len(getattr(sizes, name))
        assert g.edge_index.dtype == torch.long
        assert g.edge_attr.dtype == torch.float32
        assert g.edge_index.min() >= 0 and g.edge_index.max() < num_nodes
        assert g.attack_names[0] == "Benign"
        assert g.num_features == len(splits.feature_names)
    assert splits.train.edge_label.sum() == 0
    assert num_nodes == len(built.hosts)


def test_mesmo_ip_mesmo_indice_nos_tres_splits(built, table):
    """Cada aresta aponta para os IPs certos, com a mesma lista de hosts."""
    sizes = split_indices(table, seed=1)
    hosts = np.array(built.hosts)
    for name in ("train", "val", "test"):
        g = getattr(built.splits, name)
        rows = getattr(sizes, name)
        src_ips = table.flows["IPV4_SRC_ADDR"].astype(str).to_numpy()[rows]
        dst_ips = table.flows["IPV4_DST_ADDR"].astype(str).to_numpy()[rows]
        np.testing.assert_array_equal(hosts[g.edge_index[0].numpy()], src_ips)
        np.testing.assert_array_equal(hosts[g.edge_index[1].numpy()], dst_ips)


def test_categorias_de_ataque_corretas(built, table):
    sizes = split_indices(table, seed=1)
    g = built.splits.val
    names = np.array(g.attack_names)
    expected = table.flows[ATTACK_COL].astype(str).to_numpy()[sizes.val]
    np.testing.assert_array_equal(names[g.edge_attack.numpy()], expected)
    np.testing.assert_array_equal(g.edge_label.numpy(), (expected != "Benign"))


def test_load_splits_com_cache(tmp_path):
    make_fake_csv(tmp_path)
    cfg = DataConfig(dataset=DATASET, data_dir=str(tmp_path), seed=1)
    first = load_splits(cfg)
    folder = cache_dir(cfg)
    for name in ("graphs.pt", "meta.json", "hosts.txt", "scaler.pkl"):
        assert (folder / name).is_file()
    cached = load_cache(cfg)
    assert cached is not None
    for name in ("train", "val", "test"):
        a, b = getattr(first, name), getattr(cached, name)
        assert torch.equal(a.edge_index, b.edge_index)
        assert torch.equal(a.edge_attr, b.edge_attr)
        assert a.attack_names == b.attack_names and a.num_nodes == b.num_nodes
    assert cached.feature_names == first.feature_names


def test_pastas_de_cache_diferentes_por_config(tmp_path):
    base = {"dataset": DATASET, "data_dir": str(tmp_path)}
    dirs = {
        cache_dir(DataConfig(**base, seed=1)),
        cache_dir(DataConfig(**base, seed=2)),
        cache_dir(DataConfig(**base, seed=1, fraction=0.2)),
        cache_dir(DataConfig(**base, seed=1, drop_timestamps=False)),
    }
    assert len(dirs) == 4
    assert cache_dir(DataConfig(**base, seed=1, fraction=0.2)).name == (
        f"{DATASET}_0.2_seed1"
    )


def test_grafo_funciona_com_o_amostrador_da_f1(built):
    """Integração com a F1: o NeighborhoodSampler aceita os grafos da F2."""
    sampler = NeighborhoodSampler(built.splits.train, fanout=8)
    ids = torch.arange(16)
    node_feats, src_pos, dst_pos = sampler.batch(ids)
    assert node_feats.shape[1] == built.splits.num_features
    assert src_pos.shape == dst_pos.shape == (16,)
