#!/usr/bin/env python3
"""Batch evaluate all completed groups."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from neatmem.evaluation.metrics.llm_judge import evaluate_llm_judge

def judge_group(search_file: str, output_file: str):
    with open(search_file, "r") as f:
        data = json.load(f)

    LLM_JUDGE = {}
    RESULTS = {}

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            if int(category) == 5:
                continue

            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE.setdefault(category, []).append(label)
            RESULTS.setdefault(index, []).append({
                "question": question,
                "gt_answer": gold_answer,
                "response": generated_answer,
                "category": category,
                "llm_label": label,
            })
        index += 1

    with open(output_file, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Summary
    total_correct = sum(sum(v) for v in LLM_JUDGE.values())
    total_count = sum(len(v) for v in LLM_JUDGE.values())
    overall = total_correct / total_count if total_count else 0

    print(f"\n=== {os.path.basename(os.path.dirname(search_file))} ===")
    print(f"Overall: {overall:.4f} ({total_correct}/{total_count})")
    for cat in sorted(LLM_JUDGE.keys(), key=int):
        vals = LLM_JUDGE[cat]
        print(f"  Cat{cat}: {np.mean(vals):.4f} ({sum(vals)}/{len(vals)})")

    return LLM_JUDGE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="all", help="Group name or 'all'")
    args = parser.parse_args()

    base = "/root/autodl-tmp/NeatMem/evaluation/outputs"
    groups = [d for d in os.listdir(base) if d.startswith("g") and os.path.isfile(os.path.join(base, d, "neatmem_results.json"))]
    groups.sort()

    if args.group != "all":
        groups = [g for g in groups if g == args.group]

    for g in groups:
        search_file = os.path.join(base, g, "neatmem_results.json")
        judged_file = os.path.join(base, g, "judged.json")
        if os.path.exists(judged_file):
            print(f"Skipping {g} (already judged)")
            continue
        print(f"\nJudging {g}...")
        judge_group(search_file, judged_file)


if __name__ == "__main__":
    main()
