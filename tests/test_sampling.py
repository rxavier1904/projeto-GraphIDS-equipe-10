import torch

from graphids.models.sampling import NeighborhoodSampler


def test_media_de_entrada_completa(tiny_graph):
    s = NeighborhoodSampler(tiny_graph, fanout=-1)
    attr = tiny_graph.edge_attr
    assert torch.allclose(s.full_mean[2], attr[[0, 1, 2]].mean(0))
    assert torch.allclose(s.full_mean[1], attr[[3, 5]].mean(0))
    # nó sem arestas de entrada fica com vetor nulo
    assert torch.equal(s.full_mean[3], torch.zeros(2))


def test_fanout_limita_amostra(tiny_graph):
    s = NeighborhoodSampler(tiny_graph, fanout=2)
    assert s.heavy.tolist() == [2]
    edges = s.sample_incoming(2, torch.Generator().manual_seed(0))
    assert len(edges) == 2 and set(edges.tolist()) <= {0, 1, 2}
    assert len(set(edges.tolist())) == 2  # sem reposição


def test_batch_posicoes_apontam_para_os_hosts_certos(tiny_graph):
    s = NeighborhoodSampler(tiny_graph, fanout=-1)
    ids = torch.tensor([0, 4])  # fluxos 0->2 e 1->0
    feats, src_pos, dst_pos = s.batch(ids)
    src = tiny_graph.edge_index[0, ids]
    dst = tiny_graph.edge_index[1, ids]
    assert torch.allclose(feats[src_pos], s.full_mean[src])
    assert torch.allclose(feats[dst_pos], s.full_mean[dst])


# --- Variantes de agregação (ablação, fora da Tabela 14) ---------------------

import pytest  # noqa: E402

from graphids.config import load_config  # noqa: E402
from graphids.models import GraphIDS  # noqa: E402
from graphids.models.sampling import AGG_TYPES, make_sampler  # noqa: E402


def test_max_e_mean_max(tiny_graph):
    attr = tiny_graph.edge_attr
    s = NeighborhoodSampler(tiny_graph, fanout=-1, agg_type="max")
    assert torch.allclose(s.full_agg[2], attr[[0, 1, 2]].amax(0))
    assert torch.equal(s.full_agg[3], torch.zeros(2))  # sem arestas de entrada
    mm = NeighborhoodSampler(tiny_graph, fanout=-1, agg_type="mean_max")
    assert mm.full_agg.shape == (4, 4)
    assert torch.allclose(
        mm.full_agg[2], torch.cat([attr[[0, 1, 2]].mean(0), attr[[0, 1, 2]].amax(0)])
    )


def test_sum_comprimida(tiny_graph):
    s = NeighborhoodSampler(tiny_graph, fanout=-1, agg_type="sum")
    expected = torch.log1p(tiny_graph.edge_attr[[0, 1, 2]].sum(0))
    assert torch.allclose(s.full_agg[2], expected)


def test_wmean_pesos(tiny_graph):
    w = torch.tensor([1.0, 0.0, 3.0, 1.0, 1.0, 1.0])
    s = NeighborhoodSampler(tiny_graph, fanout=-1, agg_type="wmean", edge_weight=w)
    attr = tiny_graph.edge_attr
    expected = (attr[0] * 1 + attr[2] * 3) / 4
    assert torch.allclose(s.full_agg[2], expected)
    with pytest.raises(ValueError):
        NeighborhoodSampler(tiny_graph, agg_type="wmean")


def test_fanout_usa_mesma_agregacao(tiny_graph):
    """Com fanout >= grau, a reamostragem reproduz a agregação completa."""
    for agg in ("mean", "max", "sum", "mean_max"):
        s = NeighborhoodSampler(tiny_graph, fanout=-1, agg_type=agg)
        edges = s.in_edges[int(s.ptr[2]) : int(s.ptr[3])]
        assert torch.allclose(s.aggregate(edges, 2), s.full_agg[2])


@pytest.mark.parametrize("agg", AGG_TYPES)
def test_modelo_roda_com_cada_agregacao(splits, agg):
    cfg = load_config("configs/synthetic.yaml", [f"agg_type={agg}"])
    if agg == "wmean":  # o sintético não tem colunas de bytes
        splits.feature_names[:2] = ["IN_BYTES", "OUT_BYTES"]
    sampler = make_sampler(splits.train, cfg, splits)
    model = GraphIDS.from_config(cfg, splits.num_features)
    ids = torch.arange(64)
    out = model(
        *sampler.batch(ids, torch.Generator().manual_seed(0)), cfg.ae_batch_size
    )
    assert out.scores.shape == (64,) and torch.isfinite(out.loss)


def test_agg_type_invalido():
    with pytest.raises(ValueError):
        load_config("configs/synthetic.yaml", ["agg_type=attention"])


def test_poda_exclui_arestas_da_agregacao(tiny_graph):
    keep = torch.tensor(
        [True, False, True, True, True, True]
    )  # tira a aresta 1 -> nó 2
    s = NeighborhoodSampler(tiny_graph, fanout=-1, edge_mask=keep)
    attr = tiny_graph.edge_attr
    assert torch.allclose(s.full_agg[2], attr[[0, 2]].mean(0))
    assert set(s.in_edges.tolist()) == {0, 2, 3, 4, 5}
    edges = s.in_edges[int(s.ptr[2]) : int(s.ptr[3])]
    assert set(edges.tolist()) == {0, 2}


def test_poda_via_config(splits):
    splits.feature_names[:2] = ["IN_BYTES", "OUT_BYTES"]
    cfg = load_config("configs/synthetic.yaml", ["prune_bytes_pct=50"])
    sampler = make_sampler(splits.train, cfg, splits)
    frac = sampler.in_edges.numel() / splits.train.num_edges
    assert 0.4 < frac < 0.6
    model = GraphIDS.from_config(cfg, splits.num_features)
    out = model(*sampler.batch(torch.arange(64)), cfg.ae_batch_size)
    assert torch.isfinite(out.loss)


def test_agregador_max_completo(tiny_graph):
    s = NeighborhoodSampler(tiny_graph, fanout=-1, agg_type="max")
    attr = tiny_graph.edge_attr
    assert torch.allclose(s.full_max[2], attr[[0, 1, 2]].amax(0))
    assert torch.allclose(s.full_max[1], attr[[3, 5]].amax(0))
    # nó sem arestas de entrada fica com vetor nulo
    assert torch.equal(s.full_max[3], torch.zeros(2))
    assert s.node_dim == 2


def test_agregador_rico_mean_max_logdeg(tiny_graph):
    s = NeighborhoodSampler(tiny_graph, fanout=-1, agg_type="rich")
    attr = tiny_graph.edge_attr
    # nó 2: mean [2, 3], max [4, 5], log(1 + 3)
    expected_2 = torch.cat(
        [
            attr[[0, 1, 2]].mean(0),
            attr[[0, 1, 2]].amax(0),
            torch.log1p(torch.tensor([3.0])),
        ]
    )
    assert torch.allclose(s.full_feats[2], expected_2)
    # nó 3: tudo zero
    expected_3 = torch.zeros(5)
    assert torch.allclose(s.full_feats[3], expected_3)
    assert s.node_dim == 5


def test_agregador_rico_com_fanout_pesado(tiny_graph):
    # nó 2 recebe 3 arestas; fanout=2 faz reamostragem
    s = NeighborhoodSampler(tiny_graph, fanout=2, agg_type="rich")
    gen = torch.Generator().manual_seed(42)
    feats = s.node_features(torch.tensor([2, 3]), generator=gen)
    assert feats.shape == (2, 5)
    # log-degree do nó 2 permanece log(1 + 3)
    assert torch.allclose(feats[0, 4], torch.log1p(torch.tensor(3.0)))
    # nó 3 permanece zerado
    assert torch.allclose(feats[1], torch.zeros(5))
