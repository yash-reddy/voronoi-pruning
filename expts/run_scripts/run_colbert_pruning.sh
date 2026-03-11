#!/usr/bin/env bash
set -euo pipefail

QUERIES="${QUERIES:-data/downloads/msmarco/queries.dev.tsv}"
COLLECTION="${COLLECTION:-data/downloads/msmarco/collection.tsv}"
CHECKPOINT="${CHECKPOINT:-data/downloads/colbertv2.0}"
INDEX_NAME="${INDEX_NAME:-msmarco.2bits}"
PRUNING_RATIO="${PRUNING_RATIO:-0.5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-laplace}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-msmarco.dev.2bits.${PRUNING_RATIO}}"
N_CHUNKS="${N_CHUNKS:-24}"
PRUNE_ORDER_EXTRA_ARGS="${PRUNE_ORDER_EXTRA_ARGS:-}"
PRUNE_INDEX_EXTRA_ARGS="${PRUNE_INDEX_EXTRA_ARGS:-}"

PRUNED_INDEX="${PRUNED_INDEX:-${INDEX_NAME}.pruned.${PRUNING_RATIO}}"

timestamp() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
}

echo "$(timestamp) Submitting indexing job"
INDEX_JOBID=$(sbatch --parsable expts/run_scripts/index.slurm \
  "$QUERIES" \
  "$COLLECTION" \
  "$CHECKPOINT" \
  "$INDEX_NAME" \
  "$EXPERIMENT_NAME" \
  "$OUTPUT_FILENAME")

echo "$(timestamp) Indexing job submitted with Job ID: $INDEX_JOBID"
echo "$(timestamp) Submitting pruning jobs (total chunks: $N_CHUNKS)"

PRUNE_JOBIDS=()

for ((i = 0; i < N_CHUNKS; i++)); do
  echo "$(timestamp) Submitting prune_order job for chunk $i/$((N_CHUNKS - 1))"
  # shellcheck disable=SC2086 # We want to allow word splitting in some cases here
  PRUNE_JOBID=$(
    sbatch --parsable --dependency=afterok:"$INDEX_JOBID" expts/run_scripts/prune_order.slurm \
      "$INDEX_NAME" \
      "$EXPERIMENT_NAME" \
      "$N_CHUNKS" \
      "$i" \
      $PRUNE_ORDER_EXTRA_ARGS
  )
  PRUNE_JOBIDS+=("$PRUNE_JOBID")
done

DEPENDENCY_LIST=$(
  IFS=:
  echo "${PRUNE_JOBIDS[*]}"
)
echo "$(timestamp) All pruning jobs submitted with dependency on indexing job"
echo "$(timestamp) Submitting final pruning script with dependency on all pruning jobs"

# shellcheck disable=SC2086 # We want to allow word splitting in some cases here
FINAL_JOBID=$(
  sbatch --parsable --dependency=afterok:"${DEPENDENCY_LIST}" expts/run_scripts/prune_index.slurm \
    "$QUERIES" \
    "$INDEX_NAME" \
    "$PRUNED_INDEX" \
    "$EXPERIMENT_NAME" \
    "$OUTPUT_FILENAME" \
    "$PRUNING_RATIO" \
    $PRUNE_INDEX_EXTRA_ARGS
)

echo "$(timestamp) Final pruning job submitted with Job ID: $FINAL_JOBID"
