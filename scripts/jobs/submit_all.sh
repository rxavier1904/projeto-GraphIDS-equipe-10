#!/usr/bin/env bash
# Submete as rodadas oficiais: 4 datasets do artigo × 3 seeds (entrega 3.3 da F3).
#
# Uso (a partir da raiz do repositório):
#   scripts/jobs/submit_all.sh                         # 4 datasets do artigo
#   scripts/jobs/submit_all.sh configs/NF-ToN-IoT-v3.yaml   # configs escolhidas
#
# Os parâmetros de cada job (GPU, tempo, seeds) estão em scripts/jobs/train.sbatch.
# Um job que falhar pode ser resubmetido: as seeds já concluídas são puladas.

set -euo pipefail

CONFIGS=("$@")
if [[ -z "${CONFIGS[*]:-}" ]]; then
  CONFIGS=(
    configs/NF-UNSW-NB15-v3.yaml
    configs/NF-UNSW-NB15-v2.yaml
    configs/NF-CSE-CIC-IDS2018-v3.yaml
    configs/NF-CSE-CIC-IDS2018-v2.yaml
  )
fi

mkdir -p results/logs
for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .yaml)"
  sbatch --job-name="graphids-$name" --export="ALL,CONFIG=$config" \
    scripts/jobs/train.sbatch
done
squeue -u "$USER" || true
