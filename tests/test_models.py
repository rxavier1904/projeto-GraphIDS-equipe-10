import torch

from graphids.models import GraphIDS, NeighborhoodSampler, TransformerMAE
from graphids.models.graphids import to_windows
from graphids.models.transformer_mae import random_attention_mask


def test_mascara_simetrica_com_proporcao_esperada():
    torch.manual_seed(0)
    m = random_attention_mask(512, 0.15)
    assert torch.equal(m, m.T)
    assert not m.diagonal().any()
    off_diag = m.sum().item() / (512 * 511)
    assert abs(off_diag - 0.15) < 0.01


def test_mascara_desligada_na_inferencia():
    torch.manual_seed(0)
    mae = TransformerMAE(8, 8, num_heads=2, mask_ratio=0.5).eval()
    x = torch.randn(2, 16, 8)
    assert torch.equal(mae(x), mae(x))  # sem sorteio de máscara: determinístico


def test_janelas_e_padding():
    h = torch.randn(10, 3)
    w, pad = to_windows(h, 4)
    assert w.shape == (3, 4, 3) and pad.sum() == 2
    assert torch.equal(w.view(-1, 3)[:10], h)


def test_formas_e_score_igual_ao_erro_quadratico(splits):
    sampler = NeighborhoodSampler(splits.train, fanout=64)
    model = GraphIDS(splits.num_features, 16, 8, window_size=32, mask_ratio=0.0)
    model.eval()
    ids = torch.arange(100)
    feats, src_pos, dst_pos = sampler.batch(ids)
    h = model.embed(feats, src_pos, dst_pos)
    out = model.reconstruct(h, ae_batch_size=2)
    assert h.shape == (100, 16) and out.scores.shape == (100,)
    windows, pad = to_windows(h, 32)
    recon = model.autoencoder(windows, pad).view(-1, 16)[:100]
    assert torch.allclose(out.scores, ((h - recon) ** 2).sum(-1), atol=1e-5)


def test_gradiente_chega_ao_gnn(splits):
    sampler = NeighborhoodSampler(splits.train, fanout=64)
    model = GraphIDS(splits.num_features, 16, 8, window_size=32)
    out = model(*sampler.batch(torch.arange(200)), ae_batch_size=2)
    out.loss.backward()
    assert model.encoder.fc_neigh.weight.grad.abs().sum() > 0
    assert model.autoencoder.input_projection.weight.grad.abs().sum() > 0
