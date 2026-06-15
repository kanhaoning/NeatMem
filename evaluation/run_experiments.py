"""LOCOMO Evaluation Runner — NeatMem 评测入口"""

import argparse
import os

from src.neatmem.add import NeatMemADD
from src.neatmem.search import NeatMemSearch


def main():
    parser = argparse.ArgumentParser(description="Run LOCOMO evaluation for NeatMem")
    parser.add_argument("--method", choices=["add", "search"], required=True, help="add=ingestion, search=search+answer")
    parser.add_argument("--dataset", default="dataset/locomo10.json", help="Path to LOCOMO dataset")
    parser.add_argument("--output-folder", default="results/", help="Output folder for results")
    parser.add_argument("--top-k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--rerank", choices=["on", "off"], default=None, help="Override LLM rerank per request")
    parser.add_argument("--workers", type=int, default=8, help="ThreadPool max_workers for search")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    if args.method == "add":
        print(f"[Ingestion] Loading: {args.dataset}")
        manager = NeatMemADD(data_path=args.dataset)
        manager.process_all_conversations()
        print("[Ingestion] Done")

    elif args.method == "search":
        output_file = os.path.join(args.output_folder, "neatmem_results.json")
        print(f"[Search+Answer] Loading: {args.dataset}, top_k={args.top_k}, workers={args.workers}")
        rerank_val = {"on": True, "off": False}.get(args.rerank) if args.rerank else None
        searcher = NeatMemSearch(output_path=output_file, top_k=args.top_k, rerank=rerank_val)
        searcher.process_data_file(args.dataset, max_workers=args.workers)
        print(f"[Search+Answer] Done → {output_file}")


if __name__ == "__main__":
    main()
