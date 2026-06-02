"""生成迷你评测数据集 — 1对话 + 每类N个QA + 只保留evidence涉及的session"""

import json
import random
import argparse
from collections import defaultdict


def parse_evidence_sessions(evidence_list):
    """从 evidence 列表提取涉及的 session 编号，evidence 格式如 'D1:3' 表示 session_1 turn_3"""
    sessions = set()
    for ev in evidence_list:
        if isinstance(ev, str) and ":" in ev and ev.startswith("D"):
            sessions.add(int(ev.split(":")[0][1:]))
    return sessions


def main():
    parser = argparse.ArgumentParser(description="Generate mini LOCOMO dataset")
    parser.add_argument("--input", default="dataset/locomo10.json")
    parser.add_argument("--output", default="dataset/locomo_mini.json")
    parser.add_argument("--conversations", type=int, default=1, help="Take first N conversations")
    parser.add_argument("--qa-per-category", type=int, default=2, help="QA count per category")
    parser.add_argument("--evidence-only-sessions", action="store_true",
                        help="Only keep sessions referenced by evidence")
    parser.add_argument("--qa-priority", choices=["earliest", "random"], default="earliest",
                        help="earliest: prefer QA whose evidence uses earlier sessions; random: random sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, "r") as f:
        data = json.load(f)

    # 取前 N 个对话
    data = data[:args.conversations]

    for item in data:
        qa = item["qa"]
        conv = item["conversation"]

        # 按 category 分组
        by_cat = defaultdict(list)
        for q in qa:
            by_cat[q["category"]].append(q)

        # 每 category 选 N 个 QA
        selected_qa = []
        needed_sessions = set()
        for cat in sorted(by_cat):
            pool = by_cat[cat]
            if args.qa_priority == "earliest":
                # 按 evidence 最大 session 编号排序，优先选涉及更早 session 的
                pool_sorted = sorted(pool, key=lambda q: max(parse_evidence_sessions(q.get("evidence", [])) or {0}))
                chosen = pool_sorted[:args.qa_per_category]
            else:
                chosen = random.sample(pool, min(args.qa_per_category, len(pool)))
            selected_qa.extend(chosen)
            for q in chosen:
                needed_sessions.update(parse_evidence_sessions(q.get("evidence", [])))

        item["qa"] = selected_qa

        # 只保留 evidence 涉及的 session
        if args.evidence_only_sessions and needed_sessions:
            all_session_keys = sorted(
                [k for k in conv if k.startswith("session_") and "date" not in k],
                key=lambda k: int(k.split("_")[1]),
            )
            keep_keys = set()
            for s_num in sorted(needed_sessions):
                key = f"session_{s_num}"
                if key in conv:
                    keep_keys.add(key)
                    keep_keys.add(key + "_date_time")

            remove_keys = [k for k in conv
                           if k not in keep_keys and k not in ("speaker_a", "speaker_b")]
            for k in remove_keys:
                del conv[k]

    # 统计
    total_qa = sum(len(item["qa"]) for item in data)
    total_sessions = sum(
        sum(1 for k in item["conversation"] if k.startswith("session_") and "date" not in k)
        for item in data
    )
    cat_dist = defaultdict(int)
    for item in data:
        for q in item["qa"]:
            cat_dist[q["category"]] += 1

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"输出: {args.output}")
    print(f"  {len(data)} 对话, {total_sessions} session, {total_qa} QA")
    print(f"  Category 分布: {dict(sorted(cat_dist.items()))}")


if __name__ == "__main__":
    main()
