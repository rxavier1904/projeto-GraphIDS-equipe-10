#!/usr/bin/env bash
# Ablação da função de agregação do E-GraphSAGE (fora da Tabela 14 do artigo).
#
# Submete UM job SLURM por (variante × dataset), com a seed 1 (ou SEEDS). O baseline
# (agg_type=mean) já está em results/runs/<dataset>/seed1 e não é refeito.
#
# Variantes (ver graphids/models/sampling.py):
#   wmean     média ponderada por log1p(bytes / mediana_bytes_treino)
#   max       máximo por feature
#   sum       soma comprimida com sign·log1p
#   mean_max  [média ‖ máximo]
#
# Saídas: results/runs_agg/<variante>/<dataset>/seed1/  (metrics.json, scores...)
#         results/logs/agg_<variante>/<dataset>_seed1.log
#         results/logs/slurm_graphids-agg-<variante>-<dataset>_<jobid>_seed1.out
#
# SBATCH_ARGS repassa opções ao sbatch, ex.: só CPU quando as GPUs estão cheias:
#   SBATCH_ARGS="--gres=gpu:0 --cpus-per-task=32" scripts/jobs/...
#
# Uso (na raiz do repositório, no slurm-client):
#   scripts/jobs/submit_agg.sh                       # 4 variantes × 5 datasets
#   scripts/jobs/submit_agg.sh max sum               # só algumas variantes
#   EXCLUDE=cluster-node6 scripts/jobs/submit_agg.sh # força A100 inteira
#   SEEDS=2-3 scripts/jobs/submit_agg.sh sum         # mais seeds (array do SLURM)
#
# Um job que falhar pode ser resubmetido com o mesmo comando: run_seeds.sh pula
# o que já tem metrics.json.

set -euo pipefail

VARIANTS=("$@")
[[ -n "${VARIANTS[*]:-}" ]] || VARIANTS=(wmean max sum mean_max)
CONFIGS=(
  configs/NF-UNSW-NB15-v3.yaml
  configs/NF-UNSW-NB15-v2.yaml
  configs/NF-CSE-CIC-IDS2018-v3.yaml
  configs/NF-CSE-CIC-IDS2018-v2.yaml
  configs/NF-ToN-IoT-v3.yaml
)
SBATCH_EXTRA=()
[[ -n "${EXCLUDE:-}" ]] && SBATCH_EXTRA+=(--exclude="$EXCLUDE")

mkdir -p results/logs
for agg in "${VARIANTS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    name="$(basename "$config" .yaml)"
    CONFIG="$config" \
    EXTRA_ARGS="--set agg_type=$agg" \
    OUTPUT_DIR="results/runs_agg/$agg" \
    LOG_DIR="results/logs/agg_$agg" \
      # shellcheck disable=SC2086
      sbatch ${SBATCH_ARGS:-} --export=ALL --array="${SEEDS:-1}" --time=08:00:00 \
        --job-name="graphids-agg-$agg-$name" \
        ${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"} \
        scripts/jobs/train.sbatch
  done
done
squeue -u "$USER" || true
