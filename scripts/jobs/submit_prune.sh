#!/usr/bin/env bash
# Ablação da poda da vizinhança por bytes (fora do artigo).
#
# Um job SLURM por (corte × dataset), seed 1. Fluxos com bytes abaixo do
# percentil p dos bytes do treino não entram na agregação dos hosts, mas
# continuam sendo avaliados (prune_bytes_pct em graphids/config.py).
#
# Saídas: results/runs_prune/p<pct>/<dataset>/seed1/
#
# SBATCH_ARGS repassa opções ao sbatch, ex.: só CPU quando as GPUs estão cheias:
#   SBATCH_ARGS="--gres=gpu:0 --cpus-per-task=32" scripts/jobs/...
#
# Uso (na raiz do repositório, no slurm-client):
#   scripts/jobs/submit_prune.sh            # cortes 25 e 50 × 5 datasets
#   scripts/jobs/submit_prune.sh 10 25      # cortes escolhidos

set -euo pipefail

PCTS=("$@")
[[ -n "${PCTS[*]:-}" ]] || PCTS=(25 50)
CONFIGS=(
  configs/NF-UNSW-NB15-v3.yaml
  configs/NF-UNSW-NB15-v2.yaml
  configs/NF-CSE-CIC-IDS2018-v3.yaml
  configs/NF-CSE-CIC-IDS2018-v2.yaml
  configs/NF-ToN-IoT-v3.yaml
)
mkdir -p results/logs
for pct in "${PCTS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    name="$(basename "$config" .yaml)"
    CONFIG="$config" \
    EXTRA_ARGS="--set prune_bytes_pct=$pct" \
    OUTPUT_DIR="results/runs_prune/p$pct" \
    LOG_DIR="results/logs/prune_p$pct" \
      # shellcheck disable=SC2086
      sbatch ${SBATCH_ARGS:-} --export=ALL --array=1 --time=08:00:00 \
        --job-name="graphids-prune-p$pct-$name" \
        scripts/jobs/train.sbatch
  done
done
squeue -u "$USER" || true
