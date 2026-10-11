"""Linha de comando do GraphIDS.

Exemplos:
    # treina e avalia (grava em results/runs/<dataset>/seed<s>/)
    python -m graphids.cli --config configs/NF-UNSW-NB15-v3.yaml --seed 1

    # só avalia um checkpoint já treinado
    python -m graphids.cli --config configs/NF-UNSW-NB15-v3.yaml --seed 1 --test

    # sobrescreve hiperparâmetros sem editar o YAML
    python -m graphids.cli --config configs/synthetic.yaml --set num_epochs=5
"""

from __future__ import annotations

import argparse

from graphids.config import load_config
from graphids.training.run import run_experiment


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina e avalia o GraphIDS.")
    p.add_argument("--config", help="YAML do experimento (pasta configs/)")
    p.add_argument("--seed", type=int, help="semente (sobrescreve o YAML)")
    p.add_argument("--data-dir", help="pasta com os datasets")
    p.add_argument("--output-dir", help="onde gravar os artefatos")
    p.add_argument("--device", help="auto, cpu ou cuda")
    p.add_argument("--test", action="store_true", help="só avalia o checkpoint")
    p.add_argument("--checkpoint", help="checkpoint a usar (padrão: do run)")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="CHAVE=VALOR",
        help="sobrescreve qualquer chave da config (pode repetir)",
    )
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    overrides = list(args.set)
    for key, value in [
        ("seed", args.seed),
        ("data_dir", args.data_dir),
        ("output_dir", args.output_dir),
        ("device", args.device),
    ]:
        if value is not None:
            overrides.append(f"{key}={value}")
    cfg = load_config(args.config, overrides)
    return run_experiment(cfg, test_only=args.test, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
