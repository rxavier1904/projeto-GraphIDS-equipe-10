#!/usr/bin/env bash
# Roda o GraphIDS com várias seeds em sequência (entrega 3.3 da F3).
#
# Cada seed grava em results/runs/<dataset>/seed<s>/ (contrato C2) e o log em
# results/logs/<dataset>_seed<s>.log. Uma seed que já tem metrics.json é pulada,
# então dá para rodar de novo depois de uma falha sem refazer o que terminou.
# Para refazer uma seed, apague a pasta dela.
#
# Uso:
#   scripts/run_seeds.sh <config.yaml> [seed ...] [-- argumentos extras da CLI]
#   scripts/run_seeds.sh configs/NF-UNSW-NB15-v3.yaml                 # seeds 1 2 3
#   scripts/run_seeds.sh configs/NF-UNSW-NB15-v3.yaml 1 2 3 -- --data-dir /dados
#
# Variáveis de ambiente: PYTHON (padrão: python), OUTPUT_DIR (padrão: results/runs),
# LOG_DIR (padrão: results/logs).

set -uo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,16p' "$0"
  exit 2
fi

CONFIG="$1"
shift
SEEDS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do SEEDS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")
[[ -n "${SEEDS[*]:-}" ]] || SEEDS=(1 2 3)

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-results/runs}"
DATASET="$("$PYTHON" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['dataset'])" "$CONFIG")"
LOG_DIR="${LOG_DIR:-results/logs}"
mkdir -p "$LOG_DIR"

failed=()
for seed in "${SEEDS[@]}"; do
  if [[ -f "$OUTPUT_DIR/$DATASET/seed$seed/metrics.json" ]]; then
    echo "[$(date '+%F %T')] $DATASET seed $seed: já concluída, pulando"
    continue
  fi
  log="$LOG_DIR/${DATASET}_seed${seed}.log"
  echo "[$(date '+%F %T')] $DATASET seed $seed: iniciando (log em $log)"
  start=$(date +%s)
  if "$PYTHON" -u -m graphids.cli --config "$CONFIG" --seed "$seed" \
      --output-dir "$OUTPUT_DIR" ${EXTRA[@]+"${EXTRA[@]}"} >"$log" 2>&1; then
    echo "[$(date '+%F %T')] $DATASET seed $seed: ok em $(( ($(date +%s) - start) / 60 )) min"
    tail -n 1 "$log"
  else
    echo "[$(date '+%F %T')] $DATASET seed $seed: FALHOU (veja $log)"
    tail -n 5 "$log"
    failed+=("$seed")
  fi
done

if [[ -n "${failed[*]:-}" ]]; then
  echo "Seeds com falha em $DATASET: ${failed[*]}"
  exit 1
fi
