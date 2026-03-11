#!/usr/bin/env bash
#SBATCH --job-name=colbertv2-lstl
#SBATCH --output=results/slurm_logs/%x-%j.out
#SBATCH --error=results/slurm_logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4

set -euo pipefail

module load git

QUERIES=$1
COLLECTION=$2
CHECKPOINT=$3
INDEX_NAME=$4
EXPERIMENT_NAME=$5
OUTPUT_FILENAME=$6

echo "EXECUTION START TIME: $(date)"

python -m expts.colbert_expts.index_collection \
  --queries "$QUERIES" \
  --collection "$COLLECTION" \
  --checkpoint "$CHECKPOINT" \
  --index-name "$INDEX_NAME" \
  --experiment "$EXPERIMENT_NAME" \
  --output-filename "$OUTPUT_FILENAME"

echo "COMPLETED INDEXING AT: $(date)"
