import glob

import pytest

from graphids.config import load_config


@pytest.mark.parametrize("path", sorted(glob.glob("configs/*.yaml")))
def test_configs_do_repositorio_carregam(path):
    cfg = load_config(path)
    assert cfg.nhops == 1 and cfg.window_size > 0


def test_override_converte_tipos():
    cfg = load_config(None, ["seed=7", "fraction=0.2", "detach_target=true"])
    assert cfg.seed == 7 and cfg.fraction == 0.2 and cfg.detach_target is True


def test_chave_desconhecida_gera_erro():
    with pytest.raises(KeyError):
        load_config(None, ["learnig_rate=0.1"])
