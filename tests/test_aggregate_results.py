"""Testes do scripts/aggregate_results.py com metrics.json sintéticos (F3)."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_results.py"


@pytest.fixture(scope="module")
def agg():
    """Importa o script (scripts/ não é um pacote)."""
    spec = importlib.util.spec_from_file_location("aggregate_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(runs_dir: Path, dataset: str, seed: int, pr_auc: float, f1: float):
    """Grava um metrics.json no formato do contrato C2."""
    out = runs_dir / dataset / f"seed{seed}"
    out.mkdir(parents=True)
    metrics = {
        "dataset": dataset,
        "seed": seed,
        "test_pr_auc": pr_auc,
        "test_f1_macro": f1,
        "val_best_pr_auc": pr_auc,
        "threshold": 0.5,
        "best_epoch": 10,
        "train_time_s": 600.0 * seed,
        "inference_us_per_sample": 2.0,
        "peak_gpu_mem_mb": 1000.0,
        "num_params": 570000,
        "git_commit": "abc1234",
    }
    (out / "metrics.json").write_text(json.dumps(metrics))


@pytest.fixture
def runs_dir(tmp_path):
    runs = tmp_path / "runs"
    # UNSW-v3: dentro de 2 desvios do artigo.
    write_run(runs, "NF-UNSW-NB15-v3", 1, 0.9990, 0.990)
    write_run(runs, "NF-UNSW-NB15-v3", 2, 0.9996, 0.994)
    write_run(runs, "NF-UNSW-NB15-v3", 3, 0.9999, 0.998)
    # UNSW-v2: bem abaixo do artigo (0,8116 ± 0,0367).
    write_run(runs, "NF-UNSW-NB15-v2", 1, 0.60, 0.80)
    write_run(runs, "NF-UNSW-NB15-v2", 2, 0.62, 0.82)
    # Dataset fora da Tabela 3, com uma seed só.
    write_run(runs, "NF-ToN-IoT-v3", 1, 0.95, 0.90)
    return runs


def test_load_runs(agg, runs_dir):
    runs = agg.load_runs(runs_dir)
    assert len(runs) == 6
    assert set(runs["dataset"]) == {
        "NF-UNSW-NB15-v3",
        "NF-UNSW-NB15-v2",
        "NF-ToN-IoT-v3",
    }


def test_load_runs_vazio(agg, tmp_path):
    assert agg.load_runs(tmp_path).empty


def test_reproduction_table(agg, runs_dir):
    table = agg.reproduction_table(agg.load_runs(runs_dir))
    # Ordem da Tabela 3 primeiro, datasets novos no fim.
    assert list(table.index) == ["NF-UNSW-NB15-v3", "NF-UNSW-NB15-v2", "NF-ToN-IoT-v3"]

    v3 = table.loc["NF-UNSW-NB15-v3"]
    assert v3["n_seeds"] == 3
    assert v3["seeds"] == "1 2 3"
    assert v3["pr_auc_mean"] == pytest.approx((0.9990 + 0.9996 + 0.9999) / 3)
    assert v3["pr_auc_std"] == pytest.approx(0.000458, abs=1e-6)  # amostral
    assert v3["pr_auc_paper_mean"] == 0.9998
    assert v3["pr_auc_diff"] == pytest.approx(v3["pr_auc_mean"] - 0.9998)
    assert v3["pr_auc_within_2std"]
    assert v3["f1_macro_within_2std"]

    v2 = table.loc["NF-UNSW-NB15-v2"]
    assert v2["pr_auc_diff"] == pytest.approx(0.61 - 0.8116)
    assert not v2["pr_auc_within_2std"]

    ton = table.loc["NF-ToN-IoT-v3"]
    assert ton["pr_auc_std"] == 0.0  # uma seed só
    assert math.isnan(ton["pr_auc_paper_mean"])
    assert ton["pr_auc_within_2std"] is None


def test_cost_table(agg, runs_dir):
    table = agg.cost_table(agg.load_runs(runs_dir))
    v3 = table.loc["NF-UNSW-NB15-v3"]
    assert v3["train_time_min_mean"] == pytest.approx(20.0)  # (10+20+30)/3 minutos
    assert v3["num_params_mean"] == 570000


def test_main_grava_tabelas(agg, runs_dir, tmp_path):
    out = tmp_path / "out"
    agg.main(["--runs", str(runs_dir), "--out", str(out)])
    for name in ("tabela_reproducao", "tabela_custo"):
        assert (out / f"{name}.csv").is_file()
        tex = (out / f"{name}.tex").read_text(encoding="utf-8")
        assert "\\begin{tabular}" in tex
        assert "NF-UNSW-NB15-v3" in tex
    repro = (out / "tabela_reproducao.tex").read_text(encoding="utf-8")
    assert "$0,9998 \\pm 0,0007$" in repro  # valor do artigo com vírgula
    assert "—" in repro  # NF-ToN-IoT-v3 não está no artigo


def test_main_sem_execucoes(agg, tmp_path):
    with pytest.raises(SystemExit):
        agg.main(["--runs", str(tmp_path), "--out", str(tmp_path / "out")])
