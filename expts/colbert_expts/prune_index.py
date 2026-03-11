"""Contains the experiment to evaluate pruning effectiveness for ColBERTv2.

Created using examples from ColBERTv2 example notebook : https://github.com/stanford-futuredata/ColBERT/blob/main/docs/intro.ipynb
"""

import argparse
from collections import Counter, defaultdict
import heapq
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import NamedTuple

from expts.colbert_expts.utils import (
    MAX_ERROR_METRIC,
    MEAN_ERROR_METRIC,
    VOLUME_METRIC,
    get_prune_order_files,
    log_args,
    prune_order_filename_mods,
)

sys.path.insert(0, str(Path("./expts/colbert").resolve()))

from colbert import Searcher
from colbert.data import Queries
from colbert.infra import Run, RunConfig
import numpy as np
import torch
import tqdm

logger = logging.getLogger(__name__)
MIN_INT = -(2**31)
GLOBAL_PRUNE_MODE = "global"
LOCAL_PRUNE_MODE = "local"


class CandidateTuple(NamedTuple):
    """Represents a candidate document embedding."""

    pid: int
    embedding_idx: int


class UpdateCandidate(NamedTuple):
    """Represents a candidate for an update and contains volume scores updates."""

    pid: int
    embedding_idx: int
    old_volume_score: int
    new_volume_score: int


class PIDRange(NamedTuple):
    """Represents a continuous range of passage IDs."""

    start: int
    end: int


PruningCandidate = np.dtype(
    [
        ("pid", np.int32),
        ("embedding_idx", np.int32),
    ]
)


def get_cli_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Prune a precomputed ColBERT index with a precomputed order of pruning."
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to the queries file.",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        required=True,
        help="Name of the index to use or create.",
    )
    parser.add_argument(
        "--pruned-index",
        type=str,
        required=True,
        help="Location of the new index that will be created.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Specifies the experiment name used to create the output directory.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        required=True,
        help="Name of the output file to save the results. Note that the pruned file will be saved as <output-filename>.pruned.tsv",
    )
    parser.add_argument("--non-iterative", action="store_true", help="Run in non-iterative mode.")
    parser.add_argument(
        "--pruning-ratio", type=float, required=True, help="Defines the pruning ratio to apply."
    )
    parser.add_argument(
        "--prune-mode",
        type=str,
        choices=[GLOBAL_PRUNE_MODE, LOCAL_PRUNE_MODE],
        default=GLOBAL_PRUNE_MODE,
        help="Defines the pruning mode to use. '%(LOCAL_PRUNE_MODE)s' will prune each document independently, '%(GLOBAL_PRUNE_MODE)s' will prune the index as a whole. Default: %(default)s",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=VOLUME_METRIC,
        choices=[VOLUME_METRIC, MEAN_ERROR_METRIC, MAX_ERROR_METRIC],
        help="Specifies which metric to use for pruning.",
    )
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


def run_index_search(experiment_name, index_name, queries, savefile: str):
    """Retrieve relevant documents for a set of queries using a pre-computed index."""
    with Run().context(RunConfig(experiment=experiment_name)):
        searcher = Searcher(index=index_name)
        rankings = searcher.search_all(queries, k=100)
        rankings.save(savefile)


def get_n_prunable(preserved_volumes, volume_percentage):
    """Calculates the number of vectors to prune based on the volume target."""
    assert 0 <= volume_percentage < 100, "Volume target should be between 0 and 100"
    volume_target = preserved_volumes[0] * volume_percentage / 100
    for p_idx, p_vol in enumerate(preserved_volumes):
        if p_vol < volume_target:
            return p_idx


def prune_colbert_index(index_path, new_index_path, prune_targets):
    """Prunes the colbert index using volume fractions."""
    # Figure out which files contain which embeddings
    n_chunks = len(list(index_path.glob("*.metadata.json")))

    orig_metadata = {}
    for chunk_idx in range(n_chunks):
        metadata_file = index_path / f"{chunk_idx}.metadata.json"
        with metadata_file.open("r") as f:
            file_contents = json.load(f)
        orig_metadata[chunk_idx] = file_contents

    pid_ranges = []
    for chunk_idx in range(n_chunks):
        meta_dict = orig_metadata[chunk_idx]
        pid_ranges.append(
            PIDRange(
                meta_dict["passage_offset"],
                meta_dict["passage_offset"] + meta_dict["num_passages"] - 1,
            )
        )

    pid_ranges.sort(key=lambda x: x.start)
    prune_targets = np.sort(prune_targets, order="pid")

    window_start, window_end = 0, 0
    range_tracker = 0
    idx_ranges = []
    while window_end < len(prune_targets) and range_tracker < len(pid_ranges):
        if prune_targets[window_end][0] > pid_ranges[range_tracker].end:
            idx_ranges.append((window_start, window_end - 1))
            window_start = window_end
            range_tracker += 1
        else:
            window_end += 1
    if window_start <= len(prune_targets):
        idx_ranges.append((window_start, window_end - 1))  # Add the last range

    assert window_end == len(prune_targets), "Couldn't bin all target vectors into ranges"
    pruned_docs_by_chunk = defaultdict(list)
    for chunk_id in range(n_chunks):
        if chunk_id >= len(idx_ranges) or idx_ranges[chunk_id][0] > idx_ranges[chunk_id][1]:
            # Copy files over without changes
            shutil.copy(
                index_path / f"{chunk_id}.codes.pt", new_index_path / f"{chunk_id}.codes.pt"
            )
            shutil.copy(
                index_path / f"{chunk_id}.residuals.pt", new_index_path / f"{chunk_id}.residuals.pt"
            )
            shutil.copy(
                index_path / f"doclens.{chunk_id}.json", new_index_path / f"doclens.{chunk_id}.json"
            )
            continue  # Just means that there are no embeddings to prune in this chunk

        prune_range = idx_ranges[chunk_id]
        # Open all relevant files
        doclens_file = index_path / f"doclens.{chunk_id}.json"
        with doclens_file.open("r") as f:
            doclens = json.load(f)
        metadata_file = index_path / f"{chunk_id}.metadata.json"
        with metadata_file.open("r") as f:
            file_contents = json.load(f)
        n_embeds = file_contents["num_embeddings"]
        passage_offset = file_contents["passage_offset"]

        centroids_file = index_path / f"{chunk_id}.codes.pt"
        centroids = torch.load(centroids_file)
        residuals_file = index_path / f"{chunk_id}.residuals.pt"
        residuals = torch.load(residuals_file)

        doclens_cumsum = np.cumsum([0] + doclens[:-1])
        # Convert  doc-level indices to chunk-level indices
        chunk_targets = []
        for target in prune_targets[prune_range[0] : prune_range[1] + 1]:
            chunk_targets.append(
                CandidateTuple(
                    target["pid"],
                    target["embedding_idx"]
                    + doclens_cumsum[
                        target["pid"] - passage_offset
                    ],  # embedding_idx is relative to the start of the chunk
                )
            )
        prune_embed_idxs = [i.embedding_idx for i in chunk_targets]
        all_indices = torch.arange(n_embeds)
        keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(prune_embed_idxs))]
        # remove embedding centroids and residuals
        new_centroids = centroids[keep_indices]
        torch.save(new_centroids, new_index_path / f"{chunk_id}.codes.pt")
        new_residuals = residuals[keep_indices]
        torch.save(new_residuals, new_index_path / f"{chunk_id}.residuals.pt")

        # Update counts
        tokens_removed = Counter([e.pid - passage_offset for e in chunk_targets])

        new_doclens = []
        for doc_idx, doc_len in enumerate(doclens):
            if doc_len > tokens_removed[doc_idx]:
                new_doclens.append(doc_len - tokens_removed[doc_idx])
            elif doc_len == tokens_removed[doc_idx]:
                new_doclens.append(
                    doc_len - tokens_removed[doc_idx]
                )  # To ensure indexing consistency
                pruned_docs_by_chunk[chunk_id].append(doc_idx + passage_offset)
            else:
                raise ValueError(
                    f"Document length {doc_len} is less than tokens removed {tokens_removed[doc_idx]} for doc {doc_idx} in chunk {chunk_id}."
                )

        new_doclens_file = new_index_path / f"doclens.{chunk_id}.json"
        with new_doclens_file.open("w") as f:
            json.dump(new_doclens, f)

        assert (
            sum(new_doclens) == new_centroids.shape[0]
        ), "Number of tokens assigned to centroids is not equal to total number of tokens in documents."

    # Update chunk metadata
    past_chunk_metadata = None
    new_metadata = {}
    for chunk_id in range(n_chunks):
        if chunk_id >= len(idx_ranges):
            n_pruned_embeds = 0
        else:
            n_pruned_embeds = idx_ranges[chunk_id][1] - idx_ranges[chunk_id][0]
        if past_chunk_metadata is None:
            meta_dict = {
                "passage_offset": 0,
                "num_passages": orig_metadata[chunk_id]["num_passages"]
                - len(pruned_docs_by_chunk[chunk_id]),
                "embedding_offset": 0,
                "num_embeddings": orig_metadata[chunk_id]["num_embeddings"] - n_pruned_embeds,
            }
            new_metadata[chunk_id] = meta_dict
        else:
            meta_dict = {
                "passage_offset": orig_metadata[chunk_id]["passage_offset"],
                "num_passages": orig_metadata[chunk_id]["num_passages"]
                - len(pruned_docs_by_chunk[chunk_id]),
                "embedding_offset": past_chunk_metadata["embedding_offset"]
                + past_chunk_metadata["num_embeddings"],
                "num_embeddings": orig_metadata[chunk_id]["num_embeddings"] - n_pruned_embeds,
            }
            new_metadata[chunk_id] = meta_dict
        past_chunk_metadata = meta_dict
    for chunk_id in range(n_chunks):
        new_metadata_file = new_index_path / f"{chunk_id}.metadata.json"
        with new_metadata_file.open("w") as f:
            json.dump(new_metadata[chunk_id], f)

    # Update metadata.json
    with (index_path / f"{n_chunks-1}.metadata.json").open("r") as f:
        last_chunk_metadata = json.load(f)
    with (index_path / "metadata.json").open("r") as f:
        metadata = json.load(f)
    metadata["num_embeddings"] = (
        last_chunk_metadata["embedding_offset"] + last_chunk_metadata["num_embeddings"]
    )
    metadata["avg_doclen"] = metadata["num_embeddings"] / (
        last_chunk_metadata["passage_offset"] + last_chunk_metadata["num_passages"]
    )
    with open(new_index_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    shutil.copy(index_path / "centroids.pt", new_index_path / "centroids.pt")
    shutil.copy(index_path / "avg_residual.pt", new_index_path / "avg_residual.pt")
    shutil.copy(index_path / "buckets.pt", new_index_path / "buckets.pt")
    shutil.copy(index_path / "plan.json", new_index_path / "plan.json")

    rebuild_ivf(new_index_path, n_chunks)
    pruned_doc_ids = []
    for ids in pruned_docs_by_chunk.values():
        pruned_doc_ids.extend(ids)


def update_collection(new_index_path, pruned_doc_ids):
    """Updates the collection to remove pruned documents."""
    index_metadata_file = new_index_path / "metadata.json"
    with index_metadata_file.open("r") as f:
        index_metadata = json.load(f)
    collection_file = index_metadata["config"]["collection"]
    new_collection_file = new_index_path / "updated_collection.tsv"
    with open(collection_file) as old_file, new_collection_file.open("w") as new_file:
        for line_idx, line in enumerate(old_file):
            if line_idx not in pruned_doc_ids:
                new_file.write(line)

    index_metadata["config"]["collection"] = str(new_collection_file)
    with index_metadata_file.open("w") as f:
        json.dump(index_metadata, f)

    index_plan_file = new_index_path / "plan.json"
    with index_plan_file.open("r") as f:
        index_plan = json.load(f)
    index_plan["config"]["collection"] = str(new_collection_file)
    with index_plan_file.open("w") as f:
        json.dump(index_plan, f)


def rebuild_ivf(new_index_path, n_chunks):
    """Rebuilds the IVF index for an index."""
    index_metadata_file = new_index_path / "metadata.json"
    with index_metadata_file.open("r") as f:
        index_metadata = json.load(f)
    n_centroids = index_metadata["num_partitions"]

    inverted_index = defaultdict(set)
    for chunk_id in range(n_chunks):
        metadata_file = new_index_path / f"{chunk_id}.metadata.json"
        with metadata_file.open("r") as f:
            chunk_metadata = json.load(f)
        doc_id_offset = chunk_metadata["passage_offset"]
        doclens_file = new_index_path / f"doclens.{chunk_id}.json"
        with doclens_file.open("r") as f:
            doclens = json.load(f)
        codes = torch.load(new_index_path / f"{chunk_id}.codes.pt")
        code_idx = 0
        for doc_id, doclen in enumerate(doclens, start=doc_id_offset):
            for _ in range(doclen):
                inverted_index[codes[code_idx].item()].add(doc_id)
                code_idx += 1

    ivf_lens = torch.tensor([len(inverted_index[i]) for i in range(n_centroids)])
    ivf = torch.tensor([elem for i in range(n_centroids) for elem in inverted_index[i]])

    new_ivf_file = new_index_path / "ivf.pid.pt"
    torch.save((ivf, ivf_lens), new_ivf_file)


def merge_sorted_arrays(order_arrays):
    """Perform a heap-based merge of sorted arrays."""
    total_rows = sum(arr.shape[0] for arr in order_arrays)
    num_cols = order_arrays[0].shape[1]

    # Pre-allocate output array
    merged_array = np.empty((total_rows, num_cols), dtype=order_arrays[0].dtype)

    a_iters = []
    for a_idx, arr in enumerate(order_arrays):
        a_iter = iter(arr)
        try:
            first_row = next(a_iter)
            a_iters.append((first_row[2], a_idx, first_row, a_iter))
        except StopIteration:
            pass
    heapq.heapify(a_iters)

    pos = 0
    while a_iters:
        _, a_idx, row, a_iter = heapq.heappop(a_iters)
        merged_array[pos] = row
        pos += 1
        try:
            next_row = next(a_iter)
            heapq.heappush(a_iters, (next_row[2], a_idx, next_row, a_iter))
        except StopIteration:
            pass
    return merged_array


def get_global_prune_targets(
    index_dir, is_non_iterative, pruning_ratio, metric_name, step_size, beam_size=1
):
    """Get global (corpus-level) pruning targets based on the specified pruning ratio."""
    mode_markers = prune_order_filename_mods(
        is_non_iterative, metric_name, step_size, beam_size=beam_size
    )
    global_prune_filename = f"global_prune_order{mode_markers}.npy"
    global_prune_file = index_dir / global_prune_filename
    if global_prune_file.exists():
        global_pruning_order = np.load(global_prune_file)
    else:
        prune_orders = []
        order_files = get_prune_order_files(index_dir, is_non_iterative, step_size, beam_size)
        if len(order_files) == 0:
            raise FileNotFoundError(f"Found no files specifying the pruning orders in {index_dir}")

        for order_file in tqdm.tqdm(order_files, desc="Loading pruning orders"):
            file_orders = np.load(order_file)
            prune_orders.append(file_orders)

        # Save a composite pruning order
        global_pruning_order = merge_sorted_arrays(prune_orders)
        np.save(global_prune_file, global_pruning_order)

    n_vecs = len(global_pruning_order)
    target_n = round(n_vecs * pruning_ratio)
    prune_targets = np.empty(target_n, dtype=PruningCandidate)
    prune_targets["pid"] = global_pruning_order[:target_n, 0]
    prune_targets["embedding_idx"] = global_pruning_order[:target_n, 1]
    return prune_targets


def get_local_prune_targets(index_dir, is_non_iterative, pruning_ratio, metric_name, step_size):
    """Get local (document-level) pruning targets based on the specified pruning ratio."""
    order_files = get_prune_order_files(index_dir, is_non_iterative, step_size)
    if len(order_files) == 0:
        raise FileNotFoundError(f"Found no files specifying the pruning orders in {index_dir}")

    prune_target_list = []
    n_docs_processed = 0
    for order_file in tqdm.tqdm(order_files, desc="Loading pruning orders"):
        chunk_order = np.load(order_file)
        # order by pid
        chunk_order = chunk_order[np.argsort(chunk_order[:, 0])]
        docids, doclens = np.unique(chunk_order[:, 0], return_counts=True)
        doclens_dict = dict(zip(docids.tolist(), doclens.tolist()))
        offset = 0
        while offset < len(chunk_order):
            pid = chunk_order[offset][0]
            doc_len = doclens_dict[pid]
            if doc_len + offset > len(chunk_order):
                raise ValueError(
                    f"Offset {offset} + doc_len {doc_len} exceeds length of chunk order {len(chunk_order)}"
                )
            doc_scores = chunk_order[offset : offset + doc_len]
            doc_scores = doc_scores[np.argsort(doc_scores[:, 2])]
            n_prune = round(doc_len * pruning_ratio)
            prune_target_list.extend(doc_scores[:n_prune, :2].tolist())
            offset += doc_len
            n_docs_processed += 1
            logger.info(
                "File: %s    offset: %d   docs_processed: %d   n_prune_targets: %d",
                order_file.stem,
                offset,
                n_docs_processed,
                len(prune_target_list),
            )
    placeholder_array = np.array(prune_target_list)
    prune_targets = np.empty(len(placeholder_array), dtype=PruningCandidate)
    prune_targets["pid"] = placeholder_array[:, 0]
    prune_targets["embedding_idx"] = placeholder_array[:, 1]
    return prune_targets


def main():
    """Driver function that runs all the experiments."""
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    args = get_cli_args()
    log_args(args, logger)
    logger.info("Loading queries")
    queries = Queries(path=args.queries)
    # Maybe there's a function in ColBERT's code that can replace this hardcoded path?
    index_dir = Path("experiments") / args.experiment_name / "indexes" / args.index_name
    if not index_dir.exists():
        raise FileNotFoundError(f"Found no existing index at {index_dir}")
    if args.prune_mode == GLOBAL_PRUNE_MODE:
        prune_targets = get_global_prune_targets(
            index_dir,
            args.non_iterative,
            args.pruning_ratio,
            args.metric,
            args.step_size,
            args.beam_size,
        )
    elif args.prune_mode == LOCAL_PRUNE_MODE:
        prune_targets = get_local_prune_targets(
            index_dir, args.non_iterative, args.pruning_ratio, args.metric, args.step_size
        )
    else:
        raise ValueError(
            f"Unknown pruning mode {args.prune_mode}. Supported modes: {GLOBAL_PRUNE_MODE}, {LOCAL_PRUNE_MODE}"
        )
    logger.info(" %d Pruning targets acquired. Pruning index now.", len(prune_targets))
    pruned_index_path = index_dir.parent / args.pruned_index
    pruned_index_path.mkdir(parents=True, exist_ok=True)
    prune_colbert_index(index_dir, pruned_index_path, prune_targets)
    logger.info("Completed pruning index. Running retrieval with the pruned index.")
    mode_markers = prune_order_filename_mods(
        args.non_iterative, args.step_size, beam_size=args.beam_size
    )
    run_index_search(
        args.experiment_name,
        args.pruned_index,
        queries,
        args.output_filename + f".pruned.{args.pruning_ratio}{mode_markers}.tsv",
    )
    logger.info("Completed running retrieval with the pruned index.")
    logger.info("Finished executing script.")


if __name__ == "__main__":
    main()
