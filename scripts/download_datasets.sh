#!/usr/bin/env bash
# Download / stage the datasets used by SweepJEPA.
#
# Everything lands under ./data/ (gitignored). Only TUS-REC2024 is openly
# downloadable; the thyroid datasets are gated behind registration + DUA, so this
# script documents the access path and drops a manifest template you fill in once
# the data arrives.
#
# Usage: scripts/download_datasets.sh [tusrec|all]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
mkdir -p "${DATA}"

download_tusrec() {
  # TUS-REC2024 — OPEN (Zenodo). Optically-tracked freehand forearm sweeps.
  # Trains the pose estimator R_psi. Start here — it is the only open set.
  local dest="${DATA}/tusrec2024"
  mkdir -p "${dest}"
  echo ">> TUS-REC2024 -> ${dest}"
  echo "   Visit https://zenodo.org/ (search 'TUS-REC2024') and place the"
  echo "   downloaded scans under ${dest}/ . Direct record IDs change per"
  echo "   release; fetch with e.g.:"
  echo "     zenodo_get <RECORD_ID> -o ${dest}"
  echo "   (pip install zenodo_get)"
}

stage_gated() {
  local name="$1" dest="$2" url="$3"
  mkdir -p "${dest}"
  echo ">> ${name} (gated) -> ${dest}"
  echo "   Request access at: ${url}"
  echo "   After the DUA clears, place frames under ${dest}/ and fill in the"
  echo "   manifest CSV (columns: nodule_id,patient_id,label,ti_rads,frame_dir)."
  if [ ! -f "${dest}/manifest.csv" ]; then
    echo "nodule_id,patient_id,label,ti_rads,frame_dir" > "${dest}/manifest.template.csv"
  fi
}

case "${1:-all}" in
  tusrec)
    download_tusrec
    ;;
  all)
    download_tusrec
    stage_gated "Stanford AIMI Thyroid Cine-clip" "${DATA}/stanford_aimi_thyroid" \
      "https://aimi.stanford.edu/datasets/thyroid-ultrasound-cine-clip"
    stage_gated "ThyroidXL (MICCAI 2025)" "${DATA}/thyroidxl" \
      "https://www.miccai.org/ (see ThyroidXL challenge page)"
    ;;
  *)
    echo "usage: $0 [tusrec|all]" >&2
    exit 1
    ;;
esac

echo "Done. Downloaded/staged data lives under ${DATA} (gitignored)."
