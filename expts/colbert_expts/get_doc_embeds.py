"""Sample queries and documents, encode with ColBERT, and store embeddings + input lengths."""

import argparse
import json
import logging
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path("./expts/colbert").resolve()))

from colbert.data import Collection
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import Run, RunConfig
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

from expts.colbert_expts.utils import log_args

logger = logging.getLogger(__name__)


def get_cli_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample documents from collection, encode them, and store embeddings & input lengths."
    )
    parser.add_argument(
        "--collection", type=str, required=True, help="Path to the document collection."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Number of documents to sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save embeddings and lengths.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to trained ColBERT model checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def sample_documents(collection, num_samples, rng):
    """Randomly sample document sentences until num_samples is reached (or exhausted).

    Returns a list of sentences.
    """
    doc_ids = list(range(len(collection)))
    doc_texts = [collection[doc_id] for doc_id in rng.sample(doc_ids, num_samples)]
    return doc_texts


def main():
    """Main function to sample, encode, and save embeddings and lengths."""
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    args = get_cli_args()
    log_args(args, logger)
    checkpoint_path = args.checkpoint_path

    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading collection...")
    collection = Collection(path=args.collection)

    sampled_documents = sample_documents(collection, args.num_samples, rng)
    logger.info("Sampled %d documents.", len(sampled_documents))
    logger.info("Starting encoding...")

    with Run().context(RunConfig(nranks=1, experiment="PLACEHOLDER")):
        # We should ideally never be using nbits
        config = ColBERTConfig(doc_maxlen=300, nbits=2)
        checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint_path)
        updated_config = ColBERTConfig.from_existing(checkpoint_config, config, Run().config)
        updated_config.configure(checkpoint=checkpoint_path)
        ckpt_model = Checkpoint(updated_config.checkpoint, colbert_config=updated_config)
        ckpt_model.cuda()  # Failing this is better than silently running on CPU
        encoder = CollectionEncoder(updated_config, ckpt_model)
        embs, doclens = encoder.encode_passages(sampled_documents)

    with (output_dir / "doclens.json").open("w") as fp:
        fp.write(json.dumps(doclens, ensure_ascii=False) + "\n")

    torch.save(embs, output_dir / "doc_embeddings.pt")
    logger.info("Saved embeddings successfully.")


if __name__ == "__main__":
    main()
