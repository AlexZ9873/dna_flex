#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
OUT="data/raw/hg38.fa.gz"

echo "[download] $URL"
curl -L "$URL" -o "$OUT"

echo "[done] saved to $OUT"
echo "[next] unzip with: gunzip -k data/raw/hg38.fa.gz"
