"""Contains the experiment to evaluate pruning effectiveness for ColBERTv2.

Created using examples from ColBERTv2 example notebook : https://github.com/stanford-futuredata/ColBERT/blob/main/docs/intro.ipynb
"""

import argparse
import bisect
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path("./expts/colbert").resolve()))

from colbert.search.index_storage import IndexScorer
import numpy as np
import torch
import tqdm

from expts.colbert_expts.utils import (
    MAX_BATCH_SIZE,
    MAX_SCORE,
    PRUNE_ORDER_BASENAME,
    log_args,
    prune_order_filename_mods,
)
from vvp.utils import get_prune_targets

logger = logging.getLogger(__name__)


def get_cli_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Compute doc-level pruning orders.")
    parser.add_argument(
        "--index-name",
        type=str,
        required=True,
        help="Name of the index to use or create.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Specifies the experiment name used to create the output directory.",
    )
    parser.add_argument(
        "--n-chunks", type=int, help="Specifies how many chunks the dataset is to be divided into."
    )
    parser.add_argument(
        "--chunk-idx", type=int, help="Specifies which chunk the script is supposed to process."
    )
    parser.add_argument("--non-iterative", action="store_true", help="Run in non-iterative mode.")
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="Number of tokens to prune in every step. Default: %(default)s.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Size of the beam to use for searching pruning targets. Default: %(default)s.",
    )

    return parser.parse_args()


def smart_chunking(sorted_doclen_tuples, n_chunks, chunk_idx):
    """Divides the dataset into chunks with approximately equal computational cost."""
    costs = [i[1] for i in sorted_doclen_tuples]
    len_cumsums = np.cumsum(costs)
    total_cost = len_cumsums[-1]
    target_cost_per_chunk = total_cost / n_chunks
    chunk_start_cost = chunk_idx * target_cost_per_chunk
    chunk_end_cost = (chunk_idx + 1) * target_cost_per_chunk
    start_idx = bisect.bisect_left(len_cumsums, chunk_start_cost)
    end_idx = bisect.bisect_left(len_cumsums, chunk_end_cost)
    # Edge case handling
    if chunk_idx == n_chunks - 1 and end_idx != len(costs):
        end_idx = len(costs)
    return (start_idx, end_idx)


def main():
    """Driver function that runs all the experiments."""
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    args = get_cli_args()
    n_chunks, chunk_idx = args.n_chunks, args.chunk_idx
    if (n_chunks is None) ^ (chunk_idx is None):
        raise ValueError(
            "Either both n_chunks and chunk_idx should be specified or both should be unspecified."
        )
    num_gpus = torch.cuda.device_count()
    log_args(args, logger)
    # IndexScorer contains the optimized decompressing function.
    index_dir = Path("experiments") / args.experiment_name / "indexes" / args.index_name
    logger.info("Loading saved index.")
    sampled_points_path = Path(index_dir) / "sampled_points.pt"
    mode_markers = prune_order_filename_mods(
        args.non_iterative, args.step_size, args.chunk_idx, args.beam_size
    )
    pruning_orders_filepath = Path(index_dir) / f"{PRUNE_ORDER_BASENAME}{mode_markers}.npy"

    if pruning_orders_filepath.exists():
        logger.warning(
            "Found existing chunk file `%s`. Skipping the creation of chunk file.",
            pruning_orders_filepath,
        )
    else:
        sampled_points = torch.load(sampled_points_path)
        sampled_points = sampled_points.to("cuda")
        index_scorer = IndexScorer(index_dir, use_gpu=num_gpus > 0)
        doclens = index_scorer.doclens

        # Sorting helps reduce padding overhead a bit.
        sorted_lentuples = sorted(enumerate(doclens.tolist()), key=lambda x: x[1])
        chunk_start, chunk_end = 0, len(sorted_lentuples)
        if chunk_idx is not None:
            # Chunk based on expected number of operations, instead of number of docs.
            chunk_start, chunk_end = smart_chunking(sorted_lentuples, n_chunks, chunk_idx)
            logger.info("chunk start: %d", chunk_start)
            logger.info("chunk end: %d", chunk_end)
            logger.info("chunk size: %d", chunk_end - chunk_start)

        pruning_orders = []
        # Batching even after chunking to make this work for cases where large chunk sizes are specified
        for bat_start in tqdm.tqdm(
            range(chunk_start, chunk_end, MAX_BATCH_SIZE),
            desc="Getting pruning orders for documents",
        ):
            bat_end = min(bat_start + MAX_BATCH_SIZE, chunk_end)
            bat_pids = [e[0] for e in sorted_lentuples[bat_start:bat_end]]
            pids_tensor = torch.tensor(bat_pids, device="cuda")
            embeddings, bat_doclens = index_scorer.lookup_pids(pids_tensor)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            max_doclen = max(bat_doclens)
            mask = torch.zeros((len(bat_doclens), max_doclen), dtype=torch.bool, device="cuda")
            padded_matrix = torch.full(
                (len(bat_doclens), max_doclen, embeddings.shape[1]),
                float("-inf"),
                device="cuda",
                dtype=sampled_points.dtype,
            )
            offset = 0
            for doc_idx, doclen in enumerate(bat_doclens):
                # To ensure padding is ignored in dot product maximization
                padded_matrix[doc_idx, :doclen] = embeddings[offset : offset + doclen]
                mask[doc_idx, doclen:] = True
                offset += doclen
            padded_matrix[mask] = float("-inf")

            prune_targets, prune_scores = get_prune_targets(
                padded_matrix,
                mask,
                sampled_points,
                step_size=args.step_size,
                iterative=not args.non_iterative,
                beam_size=args.beam_size,
            )
            # Map local passage indices to passage IDs
            pidx_to_pid_map = dict(enumerate(bat_pids))
            pid_lookup = torch.zeros(max(pidx_to_pid_map.keys()) + 1, dtype=torch.long)
            for pidx, pid in pidx_to_pid_map.items():
                pid_lookup[pidx] = pid
            pid_lookup = pid_lookup.to(prune_targets.device)
            prune_targets[:, 0] = pid_lookup[prune_targets[:, 0].long()]
            combined_tensor = torch.cat(
                [prune_targets.to(torch.float32), prune_scores.unsqueeze(1)], dim=1
            ).cpu()
            pruning_orders.append(combined_tensor)
        pruning_order_array = torch.cat(pruning_orders, dim=0).numpy()
        # Remove padding
        filtered_order = pruning_order_array[pruning_order_array[:, 2] != MAX_SCORE]
        sorted_order = filtered_order[filtered_order[:, 2].argsort()]
        np.save(pruning_orders_filepath, sorted_order)
    logger.info("Completed running retrieval with the pruned index.")
    logger.info("Finished executing script.")


if __name__ == "__main__":
    main()
