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


def test_run_name_e_run_dir():
    from pathlib import Path

    from graphids.training.run import run_dir

    cfg = load_config(
        None,
        ["dataset=NF-ToN-IoT-v3", "run_name=NF-ToN-IoT-v3-rich", "seed=1"],
    )
    assert cfg.dataset == "NF-ToN-IoT-v3"
    assert cfg.run_name == "NF-ToN-IoT-v3-rich"
    assert run_dir(cfg) == Path("results/runs/NF-ToN-IoT-v3-rich/seed1")
    assert cfg.data_config().dataset == "NF-ToN-IoT-v3"
