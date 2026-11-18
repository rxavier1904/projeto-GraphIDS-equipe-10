#!/usr/bin/env bash
# Job SLURM do GraphIDS: um array job com uma tarefa por seed (entrega 3.3 da F3).
#
# Configurado para o cluster SLURM do CIn (ver docs/SERVIDOR.md): partição
# short-simple (máx. 2 dias; a debug recusa jobs por QoS) e 1 GPU. Sem
# --exclude, o job pode cair numa fatia MIG de 10 GB do cluster-node6; para
# exigir A100 inteira, submeta com: sbatch --exclude=cluster-node6 ...
#
# Uso (a partir da raiz do repositório):
#   sbatch --export=ALL,CONFIG=configs/NF-UNSW-NB15-v3.yaml scripts/jobs/train.sbatch
#   sbatch --array=1 --export=ALL,CONFIG=... scripts/jobs/train.sbatch    # só seed 1
#   scripts/jobs/submit_all.sh                                            # 4 datasets
#
# Variáveis: CONFIG (obrigatória), DATA_DIR (padrão: data), VENV (padrão: .venv),
# EXTRA_ARGS (argumentos extras da CLI, ex.: "--set num_epochs=5").
#
#SBATCH --job-name=graphids
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm_%x_%A_seed%a.out
#SBATCH --partition=short-simple

set -euo pipefail

: "${CONFIG:?defina CONFIG, ex.: --export=ALL,CONFIG=configs/NF-UNSW-NB15-v3.yaml}"
DATA_DIR="${DATA_DIR:-data}"
VENV="${VENV:-.venv}"
SEED="${SLURM_ARRAY_TASK_ID:-1}"

cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "host $(hostname) | job ${SLURM_JOB_ID:-local} | seed $SEED | config $CONFIG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# run_seeds.sh pula a seed se ela já tiver metrics.json e grava o log da execução.
# shellcheck disable=SC2086
PYTHON=python scripts/run_seeds.sh "$CONFIG" "$SEED" -- --data-dir "$DATA_DIR" ${EXTRA_ARGS:-}
