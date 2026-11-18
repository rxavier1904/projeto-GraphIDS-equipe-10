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
