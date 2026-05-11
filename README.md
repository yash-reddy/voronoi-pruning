# Voronoi Pruning for Late-Interaction Retrieval

> 🎉 **Accepted to SIGIR 2026** — *49th International ACM SIGIR Conference on Research and Development in Information Retrieval*, Melbourne, Australia, July 20–24, 2026.

[![arXiv](https://img.shields.io/badge/arXiv-2603.09933-b31b1b)](https://arxiv.org/abs/2603.09933)

**Voronoi Pruning** is a post-hoc, training-free method for reducing the size of indexes produced by late-interaction retrievers while preserving retrieval quality. It ranks document tokens by their *Pruning Error*, a measure of the expected max-similarity error induced by pruning a token, and discards the least influential ones. At a 50% token budget it retains 98% of unpruned ColBERTv2 performance, and remains competitive at pruning ratios where existing baselines degrade sharply.

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Pruning ColBERTv2 Indices](#pruning-colbertv2-indices)
- [Key Results](#key-results)
- [Citation](#citation)
- [Contact](#contact)

---

## Installation

Create a Python 3.13 environment and install dependencies:

```shell
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes PyTorch. Depending on your CUDA version, you may need to install it separately - see the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

For development dependencies (linting, testing, etc.):

```shell
pip install -r requirements-dev.txt
```

---

## Quickstart

The `vvp` module provides self-contained implementations of the core pruning functions. The snippet below computes pruning targets for a small batch of document vectors, no index or retrieval stack required:

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

# True where a position is padding (not a real token)
vvp_mask = torch.tensor([
    [False, False, True],   # doc_1 has 2 tokens; third slot is padding
    [False, False, False],  # doc_2 has 3 tokens
])

# Sample query directions uniformly from the unit sphere
sample_points = sample_in_unit_ball(n=2, num_points=10_000)

# Returns token indices sorted by ascending Voronoi error (safest to prune first)
prune_targets, prune_scores = get_prune_targets(padded_matrix, vvp_mask, sample_points)

print(prune_targets)   # e.g. tensor([[1, 0, -1], [1, 2, 0]])
print(prune_scores)    # corresponding mean Voronoi errors
```

`prune_scores` are directly interpretable: lower values indicate tokens whose removal causes minimal expected drop in retrieval scores. Tokens with score 0.0 are losslessly prunable.

---

## Pruning ColBERTv2 Indices

The `expts/` directory contains the full pipeline used in the paper — from index construction to downstream evaluation:

| Step | Description |
|---|---|
| Index construction | Build ColBERTv2 indices for MS MARCO and BEIR |
| Pruning order computation | Rank all tokens by Voronoi removal cost |
| Index pruning | Apply pruning at any target token-budget ratio |
| Evaluation | Score nDCG@10 / MRR@10 on retrieval benchmarks |

The scripts correspond directly to the experimental configurations reported in the paper and can be applied to any existing ColBERTv2 index. Detailed setup and execution instructions are in [`expts/README.md`](expts/README.md).

---

## Key Results

At a **50% token budget**, Voronoi Pruning achieves **MRR@10 = 38.9** on MS MARCO, preserving **98% of unpruned ColBERTv2-e2e performance** while outperforming all learning-free baselines (IDF pruning: 32.6, attention-score pruning: 36.0, first-*p* pruning: 37.7).

At extreme pruning ratios (6% tokens remaining), Voronoi Pruning maintains **nDCG@10 = 0.67** on TREC-DL, versus **0.46** for the next best baseline under identical conditions.

Mean Error (ME), the average Voronoi-weighted retrieval degradation, tracks nDCG@10 with *R*² = 0.99, providing a query-free proxy for selecting token budgets without running multiple retrieval evaluations.

---

## Citation

```
@misc{kankanampati2026voronoicellformulationprincipled,
      title={A Voronoi Cell Formulation for Principled Token Pruning in Late-Interaction Retrieval Models}, 
      author={Yash Kankanampati and Yuxuan Zong and Nadi Tomeh and Benjamin Piwowarski and Joseph Le Roux},
      year={2026},
      eprint={2603.09933},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2603.09933}, 
}
```

---

## Contact

For questions regarding the implementation or experiments, please contact Yash or his academic supervisors Nadi and Joseph at `{lastname}@lipn.fr`, or Yuxuan or his academic supervisor Benjamin at `(name).(surname)@isir.upmc.fr`.