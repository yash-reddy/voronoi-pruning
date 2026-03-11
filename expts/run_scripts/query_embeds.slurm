#!/usr/bin/env bash
#SBATCH --job-name=colbert-query-embeds
#SBATCH --output=results/slurm_logs/%x-%j.out
#SBATCH --error=results/slurm_logs/%x-%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1

set -euo pipefail

module load git

QUERIES=$1
COLLECTION=$2
INDEX_NAME=$3
EXPERIMENT_NAME=$4
OUTPUT_DIR=$5

python -m expts.colbert_expts.get_query_dists \
  --queries "$QUERIES" \
  --collection "$COLLECTION" \
  --index-name "$INDEX_NAME" \
  --experiment-name "$EXPERIMENT_NAME" \
  --output-dir "$OUTPUT_DIR"
