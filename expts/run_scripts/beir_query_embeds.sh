#!/usr/bin/env bash
set -euo pipefail

SLURM_SCRIPT="expts/run_scripts/query_embeds.slurm"

BASE_QUERY_DIR="data/downloads/beir"
BASE_OUTPUT_DIR="results/beir_sample_embeds"

EXPERIMENT_NAME="laplace"

# String-based config (dataset names)
DATASETS="fiqa nfcorpus scifact"

for DATASET in $DATASETS; do
  echo "Launching job for: $DATASET"

  QUERIES="${BASE_QUERY_DIR}/${DATASET}/train/queries.tsv"
  COLLECTION="${BASE_QUERY_DIR}/${DATASET}/train/docs.tsv"
  INDEX_NAME="${DATASET}.2bits"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATASET}"

  sbatch "$SLURM_SCRIPT" \
    "$QUERIES" \
    "$COLLECTION" \
    "$INDEX_NAME" \
    "$EXPERIMENT_NAME" \
    "$OUTPUT_DIR"

done
