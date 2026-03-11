# Voronoi Pruning

Voronoi Pruning is a principled token pruning method for late-interaction retrieval models. It formulates token importance via Voronoi cell removal error under a Monte Carlo approximation, providing a geometric criterion for identifying redundant and low-importance token embeddings.

This repository contains the official implementation of:

[A Voronoi Cell Formulation for Principled Token Pruning in Late-Interaction Retrieval Models](https://arxiv.org/abs/2603.09933)

## Table of Contents

- [Installation](#installation)
- [Example Usage](#example-usage)
- [Pruning ColBERTv2 Indices](#pruning-colbertv2-indices)
- [Contact](#contact)
- [Citation](#citation)

## Installation

Install the required packages using the following commands in a `python 3.13` environment:

```shell
pip install -r requirements.txt
```

**NOTE:** The required packages include `pytorch`, which may have specific installation requirements based on your system configuration. Please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for detailed instructions on installing PyTorch for your specific environment.

To install additional development dependencies (recommended if you're modifying the codebase):

```shell
pip install -r requirements-dev.txt
```

## Example Usage

The `vvp` module contains lightweight implementations of the core Voronoi Pruning functions. The following script provides a quick example of how to use the `vvp` package to identify pruning targets for a set of document vectors:

```python
import numpy as np
import torch
from vvp.utils import sample_in_unit_ball, get_prune_targets

doc_1 = np.array([[1.0, 0.0], [-1.0, 0.0]])
doc_2 = np.array([[0.0, 1.0], [-0.1, 0.9], [0.1, 0.0]])

# Pad documents to the same length
padded_matrix = torch.zeros(2, 3, 2, dtype=torch.float32)
padded_matrix[0, :2] = torch.tensor(doc_1)
padded_matrix[1, :3] = torch.tensor(doc_2)

vvp_mask = torch.tensor([
    [False, False, True],   # doc_1 has 2 tokens, third is padding
    [False, False, False],  # doc_2 has 3 tokens
])

sample_points = sample_in_unit_ball(n=2, num_points=10000)
prune_targets, prune_scores = get_prune_targets(padded_matrix, vvp_mask, sample_points)

print(prune_targets)
print(prune_scores)
```

## Pruning ColBERTv2 Indices

The `expts` directory contains the full experimental pipeline used in the paper, including:

- Construction of ColBERTv2 indices

- Computation of Voronoi pruning orders for ColBERTv2 indices

- Pruning indices at specified target ratios

- Evaluation on downstream retrieval benchmarks

These routines can be applied directly to prune existing ColBERTv2 indices.

The provided scripts correspond directly to the experimental setups and configurations reported in the paper.

For detailed instructions on their setup and execution, please refer to the [expts/README.md](expts/README.md) file within the directory.

## Contact

For questions regarding the implementation or experiments, please contact: Yash or his academic supervisors Nadi and Joseph at {lastname}@lipn.fr, or Yuxuan or his academic supervisor Benjamin at (name).(surname)@isir.upmc.fr

## Citation

```
@misc{kankanampati2026voronoicellformulationprincipled,
      title={A Voronoi Cell Formulation for Principled Token Pruning in Late-Interaction Retrieval Models}, 
      author={Yash Kankanampati and Yuxuan Zong and Nadi Tomeh and Benjamin Piwowarksi and Joseph Le Roux},
      year={2026},
      eprint={2603.09933},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2603.09933}, 
}
```
