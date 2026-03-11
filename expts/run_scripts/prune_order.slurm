#!/usr/bin/env bash
#SBATCH --job-name=colbert-prune-order
#SBATCH --output=results/slurm_logs/%x-%j.out
#SBATCH --error=results/slurm_logs/%x-%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1

set -euo pipefail

module load git

INDEX_NAME=$1
EXPERIMENT_NAME=$2
N_CHUNKS=$3
CHUNK_IDX=$4
shift 4

# shellcheck disable=SC2068
python -m expts.colbert_expts.get_pruning_orders \
  --index-name "$INDEX_NAME" \
  --experiment-name "$EXPERIMENT_NAME" \
  --n-chunks "$N_CHUNKS" \
  --chunk-idx "$CHUNK_IDX" \
  $@
