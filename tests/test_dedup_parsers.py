"""Unit tests for the listwise dedup response parsers.

Covers the 2026-08-26 landing: Plan A raw_decode tolerance in
_parse_action_response (multi-object concatenation recovers the first
object instead of degrading to parse_error add) and the multi-target
parser _parse_action_response_mt (judgments array / bare object compat /
concatenated objects / invalid-entry filtering).

Run:  cd <repo root> && python -m pytest tests/test_dedup_parsers.py -q
"""

from neatmem.memory_add import _parse_action_response, _parse_action_response_mt


# --- _parse_action_response (single-target, Plan A tolerance) ---

def test_single_normal_object():
    r = _parse_action_response('{"action": "update", "targetId": "2", "reason": "x"}', 3)
    assert r == {"action": "update", "target_idx": 1, "reason": "x"}


def test_single_concatenated_objects_first_wins():
    raw = ('{"action": "update", "targetId": "1", "reason": "a"}\n'
           '{"action": "update", "targetId": "2", "reason": "b"}')
    r = _parse_action_response(raw, 3)
    assert r["action"] == "update" and r["target_idx"] == 0 and r["reason"] == "a"


def test_single_garbage_is_parse_error_add():
    r = _parse_action_response("not json at all", 3)
    assert r == {"action": "add", "target_idx": -1, "reason": "parse_error"}


def test_single_unknown_action_becomes_add():
    r = _parse_action_response('{"action": "merge"}', 3)
    assert r["action"] == "add"


# --- _parse_action_response_mt (multi-target) ---

def test_mt_judgments_array():
    raw = '{"judgments": [{"action": "update", "targetId": "1"}, {"action": "keep", "targetId": "2"}]}'
    js = _parse_action_response_mt(raw, 2)
    assert [(j["action"], j["target_idx"]) for j in js] == [("update", 0), ("keep", 1)]


def test_mt_bare_single_object_compat():
    js = _parse_action_response_mt('{"action": "none", "targetId": "2"}', 3)
    assert [(j["action"], j["target_idx"]) for j in js] == [("none", 1)]


def test_mt_concatenated_objects():
    raw = '{"action": "update", "targetId": "1"}{"action": "update", "targetId": "3"}'
    js = _parse_action_response_mt(raw, 3)
    assert [(j["action"], j["target_idx"]) for j in js] == [("update", 0), ("update", 2)]


def test_mt_invalid_entries_dropped():
    raw = '{"judgments": [{"action": "explode", "targetId": "1"}, ' \
          '{"action": "update", "targetId": "9"}, ' \
          '{"action": "update", "targetId": "abc"}, ' \
          '{"action": "update", "targetId": "2"}]}'
    js = _parse_action_response_mt(raw, 3)
    assert [(j["action"], j["target_idx"]) for j in js] == [("update", 1)]


def test_mt_duplicate_target_first_wins():
    raw = '{"judgments": [{"action": "update", "targetId": "1", "reason": "first"}, ' \
          '{"action": "none", "targetId": "1", "reason": "second"}]}'
    js = _parse_action_response_mt(raw, 2)
    assert len(js) == 1 and js[0]["action"] == "update" and js[0]["reason"] == "first"


def test_mt_empty_means_add():
    assert _parse_action_response_mt('{"judgments": []}', 3) == []
    assert _parse_action_response_mt("garbage", 3) == []
