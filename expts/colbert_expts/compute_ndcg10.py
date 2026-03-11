"""Script to compute nDCG@10.

Computes nDCG@10 score for a set of retrieval results using TREC-formatted qrels and run files.
"""

import argparse
from collections import defaultdict
from typing import Dict

import pytrec_eval


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels (query relevance judgments) from a TREC-formatted file.

    The file should contain a header line followed by lines in the format:
    qid docid relevance unused_field

    Arguments:
        path (str): Path to the qrels file.

    Returns:
        Dict[str, Dict[str, int]]: A nested dictionary mapping query IDs to document IDs and their relevance scores.
    """
    qrels = defaultdict(dict)
    with open(path) as f:
        next(f)
        for line in f:
            qid, docid, relevance, _ = line.strip().split()
            qrels[qid][docid] = int(relevance)
    return qrels


def load_run(path: str) -> Dict[str, Dict[str, float]]:
    """Load a retrieval run from a ColBERT/TREC-formatted file.

    Each line in the file should be in the format:
    qid docid rank score

    Arguments:
        path (str): Path to the run file.

    Returns:
        Dict[str, Dict[str, float]]: A nested dictionary mapping query IDs to document IDs and their retrieval scores.
    """
    run = defaultdict(dict)
    with open(path) as f:
        for line in f:
            qid, docid, rank, score = line.strip().split()
            run[qid][docid] = float(score)
    return run


def main(qrels_path: str, run_path: str) -> None:
    """Load qrels and run files, evaluate the run using nDCG@10, and print the mean score.

    Arguments:
        qrels_path (str): Path to the qrels file.
        run_path (str): Path to the run file.
    """
    qrels = load_qrels(qrels_path)
    run = load_run(run_path)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    results = evaluator.evaluate(run)
    ndcg10_scores = [query_measures["ndcg_cut_10"] for query_measures in results.values()]
    mean_ndcg10 = sum(ndcg10_scores) / len(ndcg10_scores) if ndcg10_scores else 0.0

    print(f"nDCG@10: {mean_ndcg10:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute nDCG@10 from qrels and ColBERT results.")
    parser.add_argument("--qrels", required=True, help="Path to qrels file")
    parser.add_argument("--run", required=True, help="Path to ColBERT run file (TREC format)")

    args = parser.parse_args()
    main(args.qrels, args.run)
