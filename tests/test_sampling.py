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
