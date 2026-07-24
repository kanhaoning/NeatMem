"""
Concurrent LLM judge for LOCOMO results — mem0-aligned variant with optional evidence.

This script aligns the judge prompt and preprocessing with the mem0 LOCOMO
benchmark runner, including the optional `--with-evidence` mode.

Also exposes `evaluate_llm_judge()` for backward compatibility with existing
scripts such as `evaluation/evals.py` and `evaluation/batch_judge.py`.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating conversational AI memory recall. "
    "Return JSON only with the format requested."
)

# Evidence pieces mirror mem0's prompts.py
_EVIDENCE_CHUNK = """## Evidence (actual conversation messages containing the answer)
{evidence_context}
"""

_EVIDENCE_RULE = """
5. **EVIDENCE SUPPORTS ANSWER**: If the evidence corroborates the generated answer, mark CORRECT — even when the generated answer diverges from the gold answer. The gold answer may be wrong or oversimplified; if the generated answer provides a more accurate or better-supported conclusion based on the evidence, that is acceptable. Use evidence only to ACCEPT answers, never to reject them more strictly.
"""

_EVIDENCE_WRONG_CLAUSE = " AND is not supported by evidence"

_JUDGE_TEMPLATE = """Label the generated answer as CORRECT or WRONG.
{evidence_section}
## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions and sentiments in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Relative dates ("few days before November") match specific dates in the same window. A specific date (e.g., "February 2020") that is consistent with a vague reference (e.g., "a few years ago" relative to 2023) is CORRECT. Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.
{evidence_rule}
5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT — do not require the exact same emotion word.

6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT — even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer{evidence_wrong_clause}
- The answer addresses a completely different topic

## Question
Question: {{question}}
Gold answer: {{gold_answer}}
Generated answer: {{generated_answer}}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


def build_judge_prompt(evidence_context=None):
    """Build the unified judge prompt, with or without evidence.

    Mirrors mem0's _build_judge_prompt but uses NeatMem's placeholder names.
    """
    if evidence_context:
        prompt = _JUDGE_TEMPLATE.format(
            evidence_section=_EVIDENCE_CHUNK.format(evidence_context=evidence_context),
            evidence_rule=_EVIDENCE_RULE,
            evidence_wrong_clause=_EVIDENCE_WRONG_CLAUSE,
        )
        # Renumber rules when evidence rule is inserted (5->6, 6->7, 7->8)
        prompt = prompt.replace("\n5. **SEMANTIC OVERLAP", "\n6. **SEMANTIC OVERLAP")
        prompt = prompt.replace("\n6. **SAME REFERENT", "\n7. **SAME REFERENT")
        prompt = prompt.replace("\n7. **FOCUS ON KNOWLEDGE", "\n8. **FOCUS ON KNOWLEDGE")
    else:
        prompt = _JUDGE_TEMPLATE.format(
            evidence_section="",
            evidence_rule="",
            evidence_wrong_clause="",
        )
    return prompt


def preprocess_answer(category, answer):
    """Mirror mem0's preprocess_answer: truncate cat3 gold answers on ';'."""
    if category == 3 and ";" in answer:
        return answer.split(";")[0].strip()
    return answer


def load_evidence_lookup(dataset_path):
    """Build lookup: (conv_idx, dia_id) -> formatted turn text.

    Mirrors mem0's load_evidence_lookup (benchmarks/locomo/run.py:201).
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for conv_idx, conv in enumerate(data):
        conversation = conv["conversation"]
        session_dates = {}
        for key in conversation:
            if key.endswith("_date_time") and key.startswith("session_"):
                session_num = key.replace("session_", "").replace("_date_time", "")
                session_dates[session_num] = conversation[key]
        for key in conversation:
            if key.startswith("session_") and not key.endswith("_date_time"):
                if not isinstance(conversation[key], list):
                    continue
                for turn in conversation[key]:
                    dia_id = turn.get("dia_id", "")
                    if dia_id:
                        speaker = turn.get("speaker", "")
                        text = turn.get("text", "")
                        dia_match = re.match(r"D(\d+):", dia_id)
                        date_suffix = ""
                        if dia_match:
                            snum = dia_match.group(1)
                            sdate = session_dates.get(snum, "")
                            if sdate:
                                date_suffix = f", said on {sdate}"
                        lookup[(conv_idx, dia_id)] = f'[{dia_id}{date_suffix}] {speaker}: "{text}"'
    return lookup


def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
        else:
            json_str = text
    return json_str


def judge(question, gold, generated, category, model, evidence_context=None):
    processed_gold = preprocess_answer(category, gold)
    prompt = build_judge_prompt(evidence_context=evidence_context)

    # Model branch: gpt-5 / o-series align with mem0 LLMClient
    # (benchmarks/common/llm_client.py:71-83) - omit temperature, no extra_body.
    # Other models (e.g. MiniMax) keep original behavior.
    m = model.lower().split("/")[-1]  # "openai/gpt-5" -> "gpt-5"
    is_gpt5_series = m.startswith(("gpt-5", "o1", "o3", "o4"))

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt.format(
                    question=question,
                    gold_answer=processed_gold,
                    generated_answer=generated,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    if is_gpt5_series:
        # mem0 alignment: gpt-5/o-series use default temperature (1), no extra_body
        pass
    else:
        kwargs["temperature"] = 0.0
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking": {"type": "adaptive"},
        }

    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(extract_json(content))
        label = parsed.get("label", "")
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        if "CORRECT" in text and "WRONG" not in text:
            label = "CORRECT"
        elif "WRONG" in text and "CORRECT" not in text:
            label = "WRONG"
        else:
            label = ""
        reasoning = "parse_failed"
    return {"label": 1 if label == "CORRECT" else 0, "reasoning": reasoning}


def evaluate_llm_judge(question, gold_answer, generated_answer, category=1, model=None):
    """Backward-compatible wrapper used by evaluation/evals.py and batch_judge.py.

    Returns 1 for CORRECT, 0 for WRONG.
    """
    if model is None:
        model = os.environ.get("LLM_MODEL", "MiniMax-M3")
    result = judge(question, gold_answer, generated_answer, category, model)
    return result["label"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--dataset", default="evaluation/dataset/locomo10.json")
    parser.add_argument("--with-evidence", action="store_true")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "MiniMax-M3"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")))
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_file = args.output_file or f"results/llm_judge_{args.input_file.split('/')[-1]}"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load evidence lookup if requested
    evidence_lookup = {}
    if args.with_evidence:
        if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"Dataset not found: {args.dataset}")
        evidence_lookup = load_evidence_lookup(args.dataset)
        print(f"Loaded evidence lookup: {len(evidence_lookup)} entries", flush=True)

    # Build flat task list (conv_id, qa) preserving original order, skipping cat 5
    tasks = []
    for conv_id, qa_list in data.items():
        for x in qa_list:
            if int(x["category"]) == 5:
                continue
            tasks.append((conv_id, x))

    print(
        f"Judging {len(tasks)} items with {args.workers} workers "
        f"(model={args.model}, with_evidence={args.with_evidence})",
        flush=True,
    )
    t_start = time.time()

    # Judge concurrently
    labels = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_idx = {}
        for i, (conv_id, x) in enumerate(tasks):
            evidence_context = None
            if args.with_evidence:
                refs = x.get("evidence", []) or []
                pieces = []
                for dia_id in refs:
                    key = (int(conv_id), dia_id)
                    if key in evidence_lookup:
                        pieces.append(evidence_lookup[key])
                if pieces:
                    evidence_context = "\n".join(pieces)

            fut = ex.submit(
                judge,
                x["question"],
                x["answer"],
                x["response"],
                x["category"],
                args.model,
                evidence_context,
            )
            future_to_idx[fut] = i

        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            labels[idx] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t_start
                print(
                    f"  [{done}/{len(tasks)}] elapsed={elapsed:.0f}s "
                    f"rate={done/(elapsed/60):.1f}/min",
                    flush=True,
                )

    wall = time.time() - t_start

    # Assemble results in original order
    judge_by_cat = defaultdict(list)
    results = defaultdict(list)
    for i, (conv_id, x) in enumerate(tasks):
        result = labels[i]
        category = x["category"]
        judge_by_cat[category].append(result["label"])
        results[i].append({
            "question": x["question"],
            "gt_answer": x["answer"],
            "response": x["response"],
            "category": category,
            "llm_label": result["label"],
            "reasoning": result.get("reasoning", ""),
        })

    # Save (single write at end — no incremental, since concurrent)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dict(results), f, indent=4, ensure_ascii=False)

    print(f"\nFinal summary:")
    total_correct = sum(sum(v) for v in judge_by_cat.values())
    total = sum(len(v) for v in judge_by_cat.values())
    print(f"Total: {total_correct}/{total} = {total_correct/total:.4f}")
    for cat, res in sorted(judge_by_cat.items()):
        print(f"  Category {cat}: {np.mean(res):.4f} ({sum(res)}/{len(res)})")
    print(f"Saved to {output_file}")
    print(f"\nWall clock: {wall:.0f}s = {wall/60:.1f}min  ({total/(wall/60):.1f} Q/min)", flush=True)


if __name__ == "__main__":
    main()
