"""Sample queries and documents, encode with ColBERT, and store embeddings + input lengths."""

import argparse
import json
import logging
from pathlib import Path
import random
import sys

from nltk import download as nltk_download
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path("./expts/colbert").resolve()))

from colbert import Searcher
from colbert.data import Collection, Queries
from colbert.infra import Run, RunConfig

from expts.colbert_expts.utils import log_args

logger = logging.getLogger(__name__)

# Ensure punkt is available
nltk_download("punkt", quiet=True)


def get_cli_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample queries & document sentences, encode them, and store embeddings & input lengths."
    )
    parser.add_argument("--queries", type=str, required=True, help="Path to the queries file.")
    parser.add_argument(
        "--collection", type=str, required=True, help="Path to the document collection."
    )
    parser.add_argument("--index-name", type=str, required=True, help="ColBERT index to use.")
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Experiment name for Run context."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Number of queries to sample, and number of document sentences to match.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save embeddings and lengths."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace tokenizer to use for length calculation (default: bert-base-uncased).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def sample_queries(queries: Queries, num_samples: int, rng: random.Random):
    """Randomly sample queries and return a list of text strings."""
    qids = list(queries.keys())
    sampled_qids = rng.sample(qids, min(num_samples, len(qids)))
    return [queries[qid] for qid in sampled_qids]


def sample_doc_sentences(collection: Collection, num_samples: int, rng: random.Random):
    """Randomly sample document sentences until num_samples is reached (or exhausted).

    Returns a list of sentences.
    """
    doc_ids = list(range(len(collection)))
    rng.shuffle(doc_ids)

    sentences = []
    for doc_id in doc_ids:
        doc_text = collection[doc_id]
        doc_sentences = sent_tokenize(doc_text)
        for s in doc_sentences:
            sentences.append(s)
            if len(sentences) >= num_samples:
                return sentences
    return sentences  # may be < num_samples if collection exhausted


def compute_lengths(texts, tokenizer):
    """Tokenize texts and return a list of input lengths."""
    encodings = tokenizer(texts, truncation=False, add_special_tokens=True)
    return [len(ids) for ids in encodings["input_ids"]]


def save_lengths(lengths, filepath: Path):
    """Save lengths as a JSON file for easy inspection."""
    with filepath.open("w") as f:
        json.dump(lengths, f)


def main():
    """Main function to sample, encode, and save embeddings and lengths."""
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    args = get_cli_args()
    log_args(args, logger)
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info("Loading queries and collection...")
    queries = Queries(path=args.queries)
    sampled_queries = sample_queries(queries, args.num_samples, rng)
    logger.info("Sampled %d queries.", len(sampled_queries))

    collection = Collection(path=args.collection)
    sampled_sentences = sample_doc_sentences(collection, len(sampled_queries), rng)
    logger.info("Sampled %d document sentences to match query count.", len(sampled_sentences))

    logger.info("Computing token lengths...")
    query_lengths = compute_lengths(sampled_queries, tokenizer)
    doc_lengths = compute_lengths(sampled_sentences, tokenizer)

    save_lengths(query_lengths, output_dir / "query_lengths.json")
    save_lengths(doc_lengths, output_dir / "doc_lengths.json")
    logger.info("Saved query and document lengths to JSON.")

    logger.info("Starting encoding...")
    with Run().context(RunConfig(experiment=args.experiment_name)):
        searcher = Searcher(index=args.index_name)
        q_embeddings = searcher.encode(sampled_queries)
        d_embeddings = searcher.encode(sampled_sentences)

    torch.save(q_embeddings, output_dir / "query_embeddings.pt")
    torch.save(d_embeddings, output_dir / "doc_embeddings.pt")
    logger.info("Saved embeddings successfully.")


if __name__ == "__main__":
    main()
