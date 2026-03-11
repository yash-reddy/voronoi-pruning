# ColBERTv2 Index Pruning Experiments

This directory contains the experimental pipeline for applying Voronoi Pruning (VP) to ColBERTv2 indices, including index construction, pruning, and retrieval evaluation.

_Unless specified otherwise, all commands are expected to be executed from the root directory of the repository._

## Hardware and Environment

All experiments were conducted on the Jean Zay HPC cluster using **NVIDIA V100** GPUs running **CUDA 11.7**.

The provided SLURM and bash scripts are configured for this environment and may require adjustment for other systems.

## Prerequisites

These instructions build upon the base environment described in the [main README](../README.md).

**1. Environment Setup** Run the following from the repo root to install experiment-specific dependencies and update the `vvp` environment:

```shell
git submodule update --init --recursive
pip install -r expts/requirements-expts.txt
conda env update --name vvp -f expts/colbert/conda_env.yml --prune

# For storing slurm logs if running on a cluster
mkdir -p results/slurm_logs
```

**2. Data Acquisition** Ensure you have downloaded the MSMARCO and BEIR datasets into your data directory before proceeding. The `download_beir.py` script can be used to download and preprocess BEIR datasets into the required format.

## Running the Experiments

The benchmarking routine is structured into three distinct phases. You can execute the entire pipeline—from initialization to evaluation—using the provided driver scripts after ensuring that the parameters in the scripts are set according to your environment and dataset paths:

```shell
# For building and pruning the MSMARCO index
bash expts/run_scripts/run_colbert_pruning.sh
# For building and pruning the BEIR indices
bash expts/run_scripts/run_beir_pruning.sh
```

This will sequentially execute the following steps:

1. **Index Creation**: Builds the initial ColBERTv2 indices for MSMARCO (or BEIR) datasets. Also evaluates the unpruned indices to establish baseline performance metrics.

2. **Generate Pruning Orders**: Computes the Voronoi Pruning scores for each token embedding in the indices, and generates pruning orders based on these scores in parallel.

3. **Prune and Evaluate Indices**: Prunes the indices to a specified pruning level (using the generated pruning orders) and evaluates the retrieval performance using the pruned indices.

## Evaluating outputs

The retrieval outputs are stored in `experiments/<EXPT_NAME>` by default. To generate evaluation metrics, use:

- [For MSMARCO] The `evaluate_msmarco.py` script provided by the ColBERTv2 repo -

  ```shell
  cd expts/colbert
  python -m  utility.evaluate.msmarco_passages \
  --qrels <QREL_FILE> \
  --ranking <GENERATED_RANKING_FILE>
  ```

- [For BEIR] The `compute_ndcg10.py` script provided in the `expts/colbert_expts` directory -

  ```shell
  python -m expts.colbert_expts.compute_ndcg10 \
  --qrels <QREL_FILE> \
  --run <GENERATED_RANKING_FILE>
  ```

## Additional Routines

- Query and document embeddings can be sampled and generated using ColBERTv2 to analyze their distributions using the `query_embeds.slurm` script.
- A random sampled of documents can be retrieved and encoded using the `get_doc_embeds.py` script for further analysis.
