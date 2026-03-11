"""Downloads BEIR datasets and saves them in MSMARCO format with ID remapping."""

import argparse
import json
import logging
from pathlib import Path

import ir_datasets

logger = logging.getLogger(__name__)

# List of BEIR dataset names as per `ir_datasets`
BEIR_DATASETS = [
    "beir/climate-fever",
    "beir/fiqa/test",
    "beir/nq",
    "beir/scidocs",
    "beir/trec-covid",
    "beir/dbpedia-entity/test",
    "beir/nfcorpus/test",
    "beir/quora/test",
    "beir/scifact/test",
    "beir/webis-touche2020/v2",
]


def get_cli_args():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Download and preprocess BEIR datasets.")
    parser.add_argument(
        "--savepath",
        type=Path,
        default=Path("data/downloads/beir"),
        help="Location of the directory to save the datasets. Default: %(default)s",
    )
    return parser.parse_args()


def main():
    """Driver function that downloads and saves BEIR datasets."""
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    args = get_cli_args()
    base_dir = args.savepath
    base_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in BEIR_DATASETS:
        logger.info("Downloading %s...", ds_name)
        dataset = ir_datasets.load(ds_name)

        ds_dir = base_dir / ds_name.replace("beir/", "")
        ds_dir.mkdir(parents=True, exist_ok=True)

        doc_id_map = {}
        query_id_map = {}

        # Save documents
        with open(ds_dir / "docs.tsv", "w", encoding="utf-8") as f_docs:
            for idx, doc in enumerate(dataset.docs_iter()):
                doc_id_map[doc.doc_id] = idx
                # Combine title and text if title exists
                text = (
                    f"{doc.title}. {doc.text}" if hasattr(doc, "title") and doc.title else doc.text
                )
                text = text.replace("\n", " ").replace("\r", " ").strip()
                f_docs_line = f"{idx}\t{text}\n"
                f_docs.write(f_docs_line)

        with open(ds_dir / "doc_id_map.json", "w", encoding="utf-8") as f:
            json.dump(doc_id_map, f)

        # Save queries
        with open(ds_dir / "queries.tsv", "w", encoding="utf-8") as f_queries:
            for idx, query in enumerate(dataset.queries_iter()):
                query_id_map[query.query_id] = idx
                f_queries.write(f"{idx}\t{query.text.strip()}\n")

        with open(ds_dir / "query_id_map.json", "w", encoding="utf-8") as f:
            json.dump(query_id_map, f)

        # Save qrels with remapped IDs
        with open(ds_dir / "qrels.tsv", "w", encoding="utf-8") as f_qrels:
            f_qrels.write("query-id\tdoc-id\trelevance\titeration\n")
            for qrel in dataset.qrels_iter():
                qid = query_id_map[qrel.query_id]
                doc_id = doc_id_map[qrel.doc_id]
                if qid is not None and doc_id is not None:
                    f_qrels.write(f"{qid}\t{doc_id}\t{qrel.relevance}\t{qrel.iteration}\n")
        logger.info("Saved %s to %s", ds_name, ds_dir)

    logger.info("All datasets downloaded and saved.")


if __name__ == "__main__":
    main()
