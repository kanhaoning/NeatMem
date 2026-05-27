"""LOCOMO 数据集采样脚本 — 生成采样后的数据集文件，评测代码零改动"""

import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser(description="Sample LOCOMO dataset for faster evaluation")
    parser.add_argument("--input", default="dataset/locomo10.json", help="Input dataset path")
    parser.add_argument("--output", default="dataset/locomo10_sampled.json", help="Output dataset path")
    parser.add_argument("--conv-sample", type=int, default=1, help="Conversation sample 1/N (0=all)")
    parser.add_argument("--qa-sample", type=int, default=1, help="QA sample 1/N (0=all)")
    parser.add_argument("--max-sessions", type=int, default=0, help="Max sessions per conversation (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, "r") as f:
        data = json.load(f)

    total_convs = len(data)
    total_qa = sum(len(item["qa"]) for item in data)
    total_sessions = sum(
        sum(1 for k in item["conversation"] if k.startswith("session_") and "date" not in k)
        for item in data
    )
    print(f"原始数据: {total_convs} 条对话, {total_sessions} 个session, {total_qa} 个QA")

    # 对话采样
    if args.conv_sample > 1:
        n = max(1, len(data) // args.conv_sample)
        data = random.sample(data, n)

    # Session 截断
    if args.max_sessions > 0:
        for item in data:
            conv = item["conversation"]
            session_keys = sorted(
                [k for k in conv if k.startswith("session_") and "date" not in k],
                key=lambda k: int(k.split("_")[1]),
            )
            keep = set(session_keys[: args.max_sessions])
            # 保留对应的 date_time key
            keep_with_dates = set()
            for k in keep:
                keep_with_dates.add(k)
                keep_with_dates.add(k + "_date_time")
            remove_keys = [k for k in conv if k not in keep_with_dates and k not in ("speaker_a", "speaker_b")]
            for k in remove_keys:
                del conv[k]

    # QA 采样
    if args.qa_sample > 1:
        for item in data:
            item["qa"] = [q for q in item["qa"] if random.random() < 1 / args.qa_sample]

    # 统计
    result_convs = len(data)
    result_qa = sum(len(item["qa"]) for item in data)
    result_sessions = sum(
        sum(1 for k in item["conversation"] if k.startswith("session_") and "date" not in k)
        for item in data
    )

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"采样后: {result_convs} 条对话, {result_sessions} 个session, {result_qa} 个QA → {args.output}")


if __name__ == "__main__":
    main()
