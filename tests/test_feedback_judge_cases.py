"""Judge case gate — the judge prompt regression set (plan §5.6 step 0).

Runs the packaged judge against the curated case set with the LIVE judge LLM
and compares verdicts to the pinned expectations. This is the prompt
regression gate: after any feedback_judge.txt change, run the full set.

Opt-in (needs a reachable judge LLM + API key):

    NEATMEM_FEEDBACK_CASE_GATE=1 python3 -m pytest tests/test_feedback_judge_cases.py

Cases with ``expected_no_window`` are window-rule cases: no_window is produced
by feedback.window without an LLM call, so they are asserted against
window slicing instead of the judge.
"""

import json
import os

import pytest

from neatmem.feedback.window import STATUS_NO_WINDOW, slice_windows

CASE_GATE = os.environ.get("NEATMEM_FEEDBACK_CASE_GATE", "").strip().lower() in {"1", "true", "yes"}
pytestmark = pytest.mark.skipif(
    not CASE_GATE,
    reason="case gate needs a live judge LLM; set NEATMEM_FEEDBACK_CASE_GATE=1",
)

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_cases():
    cases = []
    for name in ("feedback_judge_cases.jsonl", "feedback_judge_cases_edge.jsonl"):
        with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
            cases.extend(json.loads(line) for line in f if line.strip())
    return cases


CASES = _load_cases()


def _case_id(case):
    return case["case_id"]


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_judge_case(case):
    if case.get("expected_no_window"):
        # Window rule: no complete assistant turn → no_window, judge NOT
        # called. Assert against the slicing logic itself.
        outcome = slice_windows(
            [{"message_id": "u1", "role": "user", "content": case["question"]}],
            [{"id": 1, "payload": {"anchor": "u1"}}],
        )[0]
        assert outcome.status == STATUS_NO_WINDOW, f"{case['case_id']}: expected no_window"
        return

    from neatmem.feedback.judge import judge_memories, make_judge_client

    client, model = make_judge_client()
    result = judge_memories(client, model, case["question"], case["memories"], case["answer"])
    mismatches = {
        mid: f"expected={expected} got={result['verdicts'].get(mid)} "
             f"evidence={result['evidence'].get(mid, '')!r}"
        for mid, expected in case["expected"].items()
        if result["verdicts"].get(mid) != expected
    }
    assert not mismatches, f"{case['case_id']}: {mismatches}"
