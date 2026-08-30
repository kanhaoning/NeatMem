"""Demo output: stdout rendering and the --output markdown record."""

import re
from datetime import datetime
from pathlib import Path

from neatmem import __version__

# Server request-scoped log lines are tagged with the request id
# (uuid hex[:8]) by design; frames are filtered on this prefix.
REQ_LINE_RE = re.compile(r"\[[0-9a-f]{8}[ \]]")


def print_plan(case, reps: int, arm_env: dict) -> None:
    arm = " ".join(f"{k}={v}" for k, v in sorted(arm_env.items())
                   if k.startswith(("DEDUP_", "RERANK_", "ENABLE_")))
    print(f"Plan: {len(case.existing_memories)} existing memories (verbatim), "
          f"{len(case.sessions)} session(s) replayed in order, reps={reps}")
    if arm:
        print(f"Arm:  {arm}")


def _fmt_mem(mem: dict, limit: int = 0) -> str:
    text = mem.get("memory", mem.get("text", ""))
    return text[:limit] if limit else text


def print_frames(frames: list, results: dict) -> None:
    for line in frames:
        print(f"  | {line}")
    for m in results.get("results", []):
        print(f"  + add: {_fmt_mem(m, 150)}")
    for d in results.get("duplicates", []):
        score = d.get("score")
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "?"
        print(f"  ~ update ({d.get('relation', '?')}, score={score_s}): "
              f"{(d.get('old_text') or '')[:80]!r} -> {(d.get('write_text') or '')[:80]!r}")
    for m in results.get("merged", []):
        print(f"  ~ merge: {(m.get('old_text') or '')[:80]!r} + "
              f"{(m.get('new_text') or '')[:80]!r}")


def print_final_store(mems: list) -> None:
    print(f"Final store ({len(mems)} memories):")
    for i, m in enumerate(mems, 1):
        print(f"  [{i}] {_fmt_mem(m)}")


def print_compact_rep(rep: dict) -> None:
    """Folded per-rep view for reps>1: stats line, judgment lines, final store.

    Mechanism frames (search/extract timing etc.) stay folded; the outcome
    of each add and the verbatim final store are always visible.
    """
    st = rep["stats"]
    print(f"rep {rep['idx']}: final={st['final_count']} "
          f"added={st['added']} updated={st['updated']} merged={st['merged']}")
    for step in rep["steps"]:
        r = step["results"]
        for d in r.get("duplicates", []):
            score = d.get("score")
            score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "?"
            print(f"  ~ update ({d.get('relation', '?')}, score={score_s}): "
                  f"{(d.get('old_text') or '')[:80]!r} -> {(d.get('write_text') or '')[:80]!r}")
        for m in r.get("merged", []):
            print(f"  ~ merge: {(m.get('old_text') or '')[:80]!r} + "
                  f"{(m.get('new_text') or '')[:80]!r}")
    for i, m in enumerate(rep["final"], 1):
        print(f"  [{i}] {_fmt_mem(m)}")


def print_summary(reps_data: list) -> None:
    print(f"\nSummary over {len(reps_data)} reps:")
    print(f"  {'rep':>3} {'final':>5} {'added':>5} {'updated':>7} {'merged':>6}")
    for rep in reps_data:
        st = rep["stats"]
        print(f"  {rep['idx']:>3} {st['final_count']:>5} {st['added']:>5} "
              f"{st['updated']:>7} {st['merged']:>6}")
    print("Note: single runs are stochastic; use --reps N to sample repeatedly.")


def render_md(case, reps_data: list, arm_env: dict, argv: list, llm_label: str) -> str:
    """Render the run record as a markdown document."""
    today = datetime.now().strftime("%Y-%m-%d")
    cmd = "neatmem demo " + " ".join(argv)
    arm_lines = "\n".join(f"- `{k}={v}`" for k, v in sorted(arm_env.items())
                          if k.startswith(("DEDUP_", "RERANK_", "ENABLE_", "EXTRACT_")))

    parts = [
        f"# {case.name}（{today}）",
        "",
        f"> 日期: {today} / neatmem {__version__} / {len(reps_data)} reps",
        f"> LLM: {llm_label}",
        "> 显式配置（其余为默认）:",
        arm_lines or "> （无）",
        "",
        "## 说明",
        "",
        case.description or "（待补）",
        "",
        "## 输入",
        "",
    ]
    if case.existing_memories:
        parts.append(f"existing_memories（{len(case.existing_memories)} 条，逐字写入）:")
        parts.append("")
        for em in case.existing_memories:
            ts = f"（created_at={em['created_at']}）" if em.get("created_at") else ""
            parts.append(f"> {em['text']}{ts}")
        parts.append("")
    parts.append(f"sessions（{len(case.sessions)} 个，按序回放）:")
    parts.append("")
    parts.append("| Session | 消息 |")
    parts.append("|---|---|")
    for s in case.sessions:
        msgs = "<br>".join(f"[{m['role']}] {m['content']}" for m in s["messages"])
        parts.append(f"| {s['id']} | {msgs} |")
    parts.append("")

    parts.append(f"## 结果（{today}，{len(reps_data)} reps）")
    parts.append("")
    for rep in reps_data:
        parts.append(f"### rep {rep['idx']}")
        parts.append("")
        for step in rep["steps"]:
            parts.append(f"**{step['label']}**")
            parts.append("")
            for line in step["frames"]:
                parts.append(f"    {line}")
            r = step["results"]
            for m in r.get("results", []):
                parts.append(f"- + add: {_fmt_mem(m)}")
            for d in r.get("duplicates", []):
                parts.append(f"- ~ update ({d.get('relation', '?')}, score={d.get('score')}): "
                             f"{(d.get('old_text') or '')!r} -> {(d.get('write_text') or '')!r}")
            parts.append("")
        parts.append(f"终库逐字（{rep['stats']['final_count']} 条）:")
        parts.append("")
        for i, m in enumerate(rep["final"], 1):
            parts.append(f"> [{i}] {_fmt_mem(m)}")
        parts.append("")

    parts.append("## 分析")
    parts.append("")
    parts.append("（待填写）")
    parts.append("")
    parts.append("## 复现")
    parts.append("")
    parts.append("```bash")
    parts.append(cmd)
    parts.append("```")
    parts.append("")
    return "\n".join(parts)


def write_output(case, reps_data: list, arm_env: dict, argv: list, llm_label: str,
                 out_path: str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_md(case, reps_data, arm_env, argv, llm_label), encoding="utf-8")
    return path
