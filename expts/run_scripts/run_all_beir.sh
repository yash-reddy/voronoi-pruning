#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="./expts/run_scripts/beir_configs.tsv"
MAIN_SCRIPT="./expts/run_scripts/run_colbert_pruning.sh"

# Static values shared across datasets
CHECKPOINT="data/downloads/colbertv2.0"
PRUNING_RATIO="0.5"
EXPERIMENT_NAME="laplace"

# Read header and data
tail -n +2 "$CONFIG_FILE" | while IFS=$'\t' read -r DATASET QUERIES COLLECTION INDEX_NAME OUTPUT_FILENAME N_CHUNKS; do
  echo "=== Running on dataset: $DATASET ==="

  export QUERIES="$QUERIES"
  export COLLECTION="$COLLECTION"
  export CHECKPOINT="$CHECKPOINT"
  export INDEX_NAME="$INDEX_NAME"
  export PRUNING_RATIO="$PRUNING_RATIO"
  export EXPERIMENT_NAME="$EXPERIMENT_NAME"
  export OUTPUT_FILENAME="$OUTPUT_FILENAME"
  export N_CHUNKS="$N_CHUNKS"

  echo "Loaded variables:"
  echo "  DATASET=$DATASET"
  echo "  QUERIES=$QUERIES"
  echo "  COLLECTION=$COLLECTION"
  echo "  CHECKPOINT=$CHECKPOINT"
  echo "  INDEX_NAME=$INDEX_NAME"
  echo "  PRUNING_RATIO=$PRUNING_RATIO"
  echo "  EXPERIMENT_NAME=$EXPERIMENT_NAME"
  echo "  OUTPUT_FILENAME=$OUTPUT_FILENAME"
  echo "  N_CHUNKS=$N_CHUNKS"
  echo "--------------------------"

  bash "$MAIN_SCRIPT"
done
