#!/usr/bin/env bash
#SBATCH --job-name=colbert-prune-index
#SBATCH --output=results/slurm_logs/%x-%j.out
#SBATCH --error=results/slurm_logs/%x-%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4 # We need GPUs because the script also does inference after pruning
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1

set -euo pipefail

module load git

QUERIES=$1
INDEX_NAME=$2
PRUNED_INDEX=$3
EXPERIMENT_NAME=$4
OUTPUT_FILENAME=$5
PRUNE_RATIO=$6
shift 6

# shellcheck disable=SC2068
python -m expts.colbert_expts.prune_index \
  --queries "$QUERIES" \
  --index-name "$INDEX_NAME" \
  --pruned-index "$PRUNED_INDEX" \
  --experiment-name "$EXPERIMENT_NAME" \
  --output-filename "$OUTPUT_FILENAME" \
  --pruning-ratio "$PRUNE_RATIO" \
  $@
