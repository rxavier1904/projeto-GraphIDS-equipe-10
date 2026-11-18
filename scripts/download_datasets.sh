#!/usr/bin/env bash
# Baixa, descompacta e organiza os datasets NetFlow da UQ (entrega 3.2 da F3).
#
# Fonte: https://staff.itee.uq.edu.au/marius/NIDS_datasets/
#
# Resultado: <data_dir>/<dataset>/<dataset>.csv, que é o caminho esperado por
# graphids/data/loading.py, e <data_dir>/MANIFEST.tsv com o tamanho e o SHA-256
# de cada zip e de cada CSV (registrar no docs/SERVIDOR.md).
#
# Os links da UQ (rdm.uq.edu.au/files/<id>) abrem uma página HTML, não o arquivo.
# Por isso o script aceita dois caminhos:
#   1. zip já baixado: coloque o zip em <data_dir>/raw/<dataset>.zip (baixe pelo
#      navegador e copie com scp/rsync, ou passe a URL direta abaixo);
#   2. URL direta: exporte URL_<nome> (ex.: URL_NF_UNSW_NB15_v3=https://...) com
#      o link direto do zip que o navegador usa ao clicar em "Download".
#
# Uso:
#   scripts/download_datasets.sh [data_dir] [dataset ...]
#   scripts/download_datasets.sh data                       # os 5 datasets
#   scripts/download_datasets.sh data NF-UNSW-NB15-v3       # só um

set -euo pipefail

DATA_DIR="${1:-data}"
shift || true

# Datasets do artigo (v2 e v3) e o novo dataset da F4.
ALL_DATASETS=(
  NF-UNSW-NB15-v2
  NF-UNSW-NB15-v3
  NF-CSE-CIC-IDS2018-v2
  NF-CSE-CIC-IDS2018-v3
  NF-ToN-IoT-v3
)
if [[ $# -gt 0 ]]; then DATASETS=("$@"); else DATASETS=("${ALL_DATASETS[@]}"); fi

# Página de cada dataset no site da UQ (para baixar pelo navegador).
page_url() {
  case "$1" in
    NF-UNSW-NB15-v2) echo "https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848" ;;
    NF-UNSW-NB15-v3) echo "https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f" ;;
    NF-CSE-CIC-IDS2018-v2) echo "https://rdm.uq.edu.au/files/ce5161d0-ef9c-11ed-827d-e762de186848" ;;
    NF-CSE-CIC-IDS2018-v3) echo "https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22" ;;
    NF-ToN-IoT-v3) echo "https://rdm.uq.edu.au/files/343e2e8c-6e6e-4a0c-813d-a46acea1b7f4" ;;
    *) echo "" ;;
  esac
}

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

size_bytes() { wc -c <"$1" | tr -d ' '; }

mkdir -p "$DATA_DIR/raw"
MANIFEST="$DATA_DIR/MANIFEST.tsv"
[[ -f "$MANIFEST" ]] || printf 'arquivo\tbytes\tsha256\n' >"$MANIFEST"

record() {  # registra (ou atualiza) um arquivo no MANIFEST.tsv
  local file="$1" rel
  rel="${file#"$DATA_DIR"/}"
  grep -v -F "$rel	" "$MANIFEST" >"$MANIFEST.tmp" || true
  printf '%s\t%s\t%s\n' "$rel" "$(size_bytes "$file")" "$(sha256 "$file")" >>"$MANIFEST.tmp"
  mv "$MANIFEST.tmp" "$MANIFEST"
}

failed=()
for ds in "${DATASETS[@]}"; do
  echo "== $ds"
  target="$DATA_DIR/$ds/$ds.csv"
  if [[ -f "$target" ]]; then
    echo "   já existe: $target (pulando)"
    continue
  fi

  zip="$DATA_DIR/raw/$ds.zip"
  var="URL_${ds//-/_}"
  url="${!var:-}"
  if [[ ! -f "$zip" && -n "$url" ]]; then
    echo "   baixando de $url"
    curl -fL --retry 3 -o "$zip.part" "$url" && mv "$zip.part" "$zip"
  fi
  if [[ ! -f "$zip" ]]; then
    echo "   FALTA o zip. Baixe pelo navegador em $(page_url "$ds")"
    echo "   e salve como $zip (ou exporte $var com o link direto)."
    failed+=("$ds")
    continue
  fi
  # Um link errado costuma devolver uma página HTML em vez do zip.
  if ! unzip -tq "$zip" >/dev/null 2>&1; then
    echo "   ERRO: $zip não é um zip válido (página HTML?). Apague e baixe de novo."
    failed+=("$ds")
    continue
  fi
  record "$zip"

  tmp="$DATA_DIR/raw/$ds.extract"
  rm -rf "$tmp" && mkdir -p "$tmp"
  unzip -q "$zip" -d "$tmp"
  # O CSV principal é o maior .csv do zip (alguns zips trazem também um
  # arquivo de features/descrição). O site às vezes chama o CIC-IDS2018 de
  # NF-CICIDS2018-*: ele é renomeado para o nome usado nas configs.
  csv="$(find "$tmp" -type f -name '*.csv' -exec ls -S {} + | head -n 1)"
  if [[ -z "$csv" ]]; then
    echo "   ERRO: nenhum CSV dentro de $zip"
    failed+=("$ds")
    continue
  fi
  # Os zips da UQ vêm no formato BagIt, com o SHA-1 de cada arquivo em
  # manifest-sha1.txt. Se existir, confere o CSV antes de usar.
  manifest="$(find "$tmp" -type f -name 'manifest-sha1.txt' | head -n 1)"
  if [[ -n "$manifest" ]]; then
    expected="$(grep -F "$(basename "$csv")" "$manifest" | cut -d' ' -f1)"
    actual="$(shasum -a 1 "$csv" 2>/dev/null || sha1sum "$csv")"
    if [[ -n "$expected" && "${actual%% *}" != "$expected" ]]; then
      echo "   ERRO: SHA-1 do CSV não confere com o manifest-sha1.txt da UQ"
      failed+=("$ds")
      continue
    fi
    [[ -n "$expected" ]] && echo "   SHA-1 conferido com o manifest-sha1.txt da UQ"
  fi
  mkdir -p "$DATA_DIR/$ds"
  echo "   $(basename "$csv") -> $target"
  mv "$csv" "$target"
  rm -rf "$tmp"
  record "$target"
  echo "   $(( $(size_bytes "$target") / 1024 / 1024 )) MB, $(( $(wc -l <"$target") - 1 )) fluxos"
done

echo
echo "Manifesto: $MANIFEST"
column -t -s $'\t' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
if [[ -n "${failed[*]:-}" ]]; then
  echo
  echo "Pendentes: ${failed[*]}"
  exit 1
fi
