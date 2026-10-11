import json

from graphids.cli import main
from graphids.training.run import run_experiment


def test_treino_ponta_a_ponta_detecta_ataques_sinteticos(fast_cfg):
    metrics = run_experiment(fast_cfg, log=lambda _: None)
    assert metrics["test_pr_auc"] > 0.9
    assert metrics["test_f1_macro"] > 0.9


def test_artefatos_do_contrato_c2(fast_cfg, tmp_path):
    fast_cfg.num_epochs = 2
    run_experiment(fast_cfg, log=lambda _: None)
    out = tmp_path / "runs" / "synthetic" / f"seed{fast_cfg.seed}"
    for name in [
        "metrics.json",
        "scores_val.npz",
        "scores_test.npz",
        "config.yaml",
        "model.ckpt",
        "history.json",
    ]:
        assert (out / name).exists(), name
    metrics = json.loads((out / "metrics.json").read_text())
    for key in ["test_pr_auc", "test_f1_macro", "threshold", "num_params"]:
        assert key in metrics


def test_cli_treina_e_depois_so_avalia(tmp_path):
    args = [
        "--config",
        "configs/synthetic.yaml",
        "--output-dir",
        str(tmp_path),
        "--device",
        "cpu",
        "--set",
        "num_epochs=2",
    ]
    trained = main(args)
    tested = main(args + ["--test"])
    assert abs(trained["test_pr_auc"] - tested["test_pr_auc"]) < 1e-9
