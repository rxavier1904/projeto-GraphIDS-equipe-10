"""Corte por bytes sobre os scores já salvos (sem retreinar).

Para cada execução ``results/runs/<dataset>/seed<s>/`` (baseline), lê os bytes
de cada fluxo do teste no cache do dataset (``data/cache/``) e grava:

1. ``results/tabela_bytes_faixas.csv``: taxa de detecção por faixa de bytes
   (quintis do tráfego benigno do TREINO) e por categoria de ataque, mais a
   taxa de falso positivo dos benignos em cada faixa. Mostra se os erros se
   concentram em fluxos pequenos ou grandes.
2. ``results/tabela_bytes_cascata.csv``: cascata "bytes < X → benigno direto",
   com X = percentil p ∈ {1, 5, 10, 25, 50} dos bytes do treino. Os fluxos
   cortados CONTINUAM no teste como "predito benigno"; os demais usam o score
   do modelo e o threshold da validação. Também informa a fração de fluxos que
   o modelo deixaria de processar (proxy de economia na inferência).

Bytes = IN_BYTES + OUT_BYTES em valores brutos (Min-Max desfeito com o scaler).

Uso (no servidor, onde está o cache):
    python scripts/bytes_study.py [--runs results/runs] [--data-dir data]
    python scripts/bytes_study.py --seeds 1
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from graphids.config import Config
from graphids.data import load_splits
from graphids.data.cache import cache_dir

PCTS = [1, 5, 10, 25, 50]


def raw_bytes(graph, feature_names, scaler) -> np.ndarray:
    total = np.zeros(graph.num_edges)
    for col in ("IN_BYTES", "OUT_BYTES"):
        j = feature_names.index(col)
        x = graph.edge_attr[:, j].cpu().numpy().astype(np.float64)
        total += x * scaler.data_range_[j] + scaler.data_min_[j]
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--out", default="results")
    args = p.parse_args()

    bands, cascade = [], []
    for ds_dir in sorted(Path(args.runs).iterdir()):
        for seed in args.seeds:
            run = ds_dir / f"seed{seed}"
            if not (run / "scores_test.npz").exists():
                continue
            raw = yaml.safe_load((run / "config.yaml").read_text())
            cfg = Config(
                **{k: v for k, v in raw.items() if k in Config.__dataclass_fields__}
            )
            cfg.data_dir = args.data_dir
            splits = load_splits(cfg.data_config())
            # scaler.pkl vem do nosso pipeline (data/cache.py): confiável.
            with open(cache_dir(cfg.data_config()) / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            b_train = raw_bytes(splits.train, splits.feature_names, scaler)
            b_test = raw_bytes(splits.test, splits.feature_names, scaler)

            z = np.load(run / "scores_test.npz")
            thr = json.loads((run / "metrics.json").read_text())["threshold"]
            y, att, s = z["labels"], z["attacks"], z["errors"]
            names = [str(n) for n in z["attack_names"]]
            assert len(s) == len(b_test), "scores e grafo de teste fora de ordem"
            pred = s > thr
            print(f"{ds_dir.name} seed {seed}: {len(s):,} fluxos de teste")

            # 1. detecção por faixa de bytes (quintis do treino)
            edges = np.unique(np.percentile(b_train, [0, 20, 40, 60, 80, 100]))
            band = np.clip(
                np.searchsorted(edges, b_test, side="right") - 1, 0, len(edges) - 2
            )
            for k in range(len(edges) - 1):
                in_band = band == k
                label = f"Q{k + 1} [{edges[k]:.0f}, {edges[k + 1]:.0f}]"
                for i, name in enumerate(names):
                    m = in_band & (att == i)
                    if m.sum() == 0:
                        continue
                    bands.append(
                        {
                            "dataset": ds_dir.name,
                            "seed": seed,
                            "faixa": label,
                            "categoria": name,
                            "n": int(m.sum()),
                            "taxa_alerta": float(pred[m].mean()),
                        }
                    )

            # 2. cascata bytes < X -> benigno
            base = {"dataset": ds_dir.name, "seed": seed}
            for pct in [None, *PCTS]:
                if pct is None:
                    cut, x = np.zeros(len(s), bool), float("nan")
                else:
                    x = float(np.percentile(b_train, pct))
                    cut = b_test < x
                pc = pred & ~cut
                row = {
                    **base,
                    "percentil": "sem corte" if pct is None else f"p{pct}",
                    "x_bytes": x,
                    "frac_cortada": float(cut.mean()),
                    "ataques_cortados": float(cut[y == 1].mean()),
                    "f1_macro": f1_score(y, pc, average="macro", zero_division=0),
                    "deteccao": float(pc[y == 1].mean()),
                    "fpr": float(pc[y == 0].mean()),
                }
                cascade.append(row)

    out = Path(args.out)
    pd.DataFrame(bands).to_csv(out / "tabela_bytes_faixas.csv", index=False)
    df = pd.DataFrame(cascade)
    df.to_csv(out / "tabela_bytes_cascata.csv", index=False)
    with pd.option_context(
        "display.width", 200, "display.float_format", "{:.4f}".format
    ):
        print(
            df.groupby(["dataset", "percentil"], sort=False)[
                [
                    "x_bytes",
                    "frac_cortada",
                    "ataques_cortados",
                    "f1_macro",
                    "deteccao",
                    "fpr",
                ]
            ]
            .mean()
            .to_string()
        )


if __name__ == "__main__":
    main()
