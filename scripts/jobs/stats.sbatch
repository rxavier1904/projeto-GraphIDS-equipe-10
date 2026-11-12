#!/usr/bin/env bash
# Estatísticas dos datasets (entrega 3.2 da F3, apoio às seções 6 da F2 e da F4).
#
# Gera em results/estatisticas/ a tabela-resumo (fluxos, hosts, % de ataques,
# comparada à Tabela 1 do artigo) e a contagem por categoria em treino/validação/
# teste de cada dataset. Também grava o cache dos grafos em data/cache/, que as
# rodadas de treino da seed 1 reaproveitam. Só usa CPU.
#
# Uso (a partir da raiz do repositório):
#   sbatch scripts/jobs/stats.sbatch
#   sbatch --export=ALL,CONFIGS="configs/NF-ToN-IoT-v3.yaml" scripts/jobs/stats.sbatch
#
#SBATCH --job-name=graphids-stats
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/slurm_%x_%j.out
#SBATCH --partition=short-simple

set -euo pipefail

CONFIGS="${CONFIGS:-configs/NF-UNSW-NB15-v2.yaml configs/NF-UNSW-NB15-v3.yaml \
configs/NF-CSE-CIC-IDS2018-v2.yaml configs/NF-CSE-CIC-IDS2018-v3.yaml \
configs/NF-ToN-IoT-v3.yaml}"
DATA_DIR="${DATA_DIR:-data}"
VENV="${VENV:-.venv}"

cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

args=()
for c in $CONFIGS; do args+=(--config "$c"); done

echo "host $(hostname) | job ${SLURM_JOB_ID:-local} | $(date '+%F %T')"
# A seed precisa ser a mesma das rodadas oficiais (1) para o split bater.
python -m graphids.data.stats --seed 1 --data-dir "$DATA_DIR" "${args[@]}"
echo "fim $(date '+%F %T')"
