"""Processes the collection into a ColBERT index."""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path("./expts/colbert").resolve()))

from colbert import Indexer, Searcher
from colbert.data import Collection, Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
import torch

from expts.colbert_expts.utils import log_args
from vvp.utils import (
    sample_in_unit_ball,
)

logger = logging.getLogger(__name__)


def get_cli_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Index a collection of documents using ColBERT(v2)."
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to the queries file.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Path to the file containing the document collection.",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=2,
        help="Number of bits to encode each dimension (default: 2).",
    )
    parser.add_argument(
        "--doc-maxlen",
        type=int,
        default=300,
        help="Truncate passages at this many tokens (default: 300).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint directory.",
    )
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
        "--output-filename",
        type=str,
        required=True,
        help="Name of the output file to save the results. Note that the pruned file will be saved as <output-filename>.pruned.tsv",
    )
    return parser.parse_args()


def run_index_search(experiment_name, index_name, queries, savefile: str):
    """Retrieve relevant documents for a set of queries using a pre-computed index."""
    with Run().context(RunConfig(experiment=experiment_name)):
        searcher = Searcher(index=index_name)
        rankings = searcher.search_all(queries, k=100)
        rankings.save(savefile)


def create_index(args, num_gpus):
    """Create a ColBERT index for a collection of documents."""
    collection = Collection(path=args.collection)
    with Run().context(
        RunConfig(nranks=num_gpus, experiment=args.experiment_name)
    ):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=args.doc_maxlen, nbits=args.nbits)
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name=args.index_name, collection=collection, overwrite="reuse")
    return indexer


def main():
    """Driver function that runs all the experiments."""
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    args = get_cli_args()
    num_gpus = torch.cuda.device_count()
    log_args(args, logger)
    logger.info("Loading queries")
    queries = Queries(path=args.queries)
    # Maybe there's a function in ColBERT's code that can replace this hardcoded path?
    index_dir = Path("experiments") / args.experiment_name / "indexes" / args.index_name
    sampled_points_file = index_dir / "sampled_points.pt"

    if not index_dir.exists():
        logger.info("Found no existing index, creating one now.")
        _ = create_index(args, num_gpus)
        logger.info("Completed creating index. Running retrieval without pruning.")
        run_index_search(
            args.experiment_name, args.index_name, queries, args.output_filename + ".tsv"
        )
        logger.info("Completed running retrieval for the input queries without pruning.")
    else:
        logger.info("Index already exists. Skipping index creation.")

    if sampled_points_file.exists():
        logger.info("Found sampled points in the index directory, skipping point sampling.")
    else:
        sampled_points = sample_in_unit_ball(
            128,  # TODO: Turn this into a CLI arg.
            num_points=10**5,  # TODO: Turn this into a CLI arg.
            dtype=torch.float16,
            device="cpu",
            sample_on_surface=True,
        )  # unit vectors
        torch.save(sampled_points, sampled_points_file)
    logger.info("Completed creating index / checking if index already exists.")


if __name__ == "__main__":
    main()
