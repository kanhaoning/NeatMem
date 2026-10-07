"""Tests for feedback.window — injection → answer window slicing."""

from neatmem.feedback.window import (
    STATUS_MISSING_ANCHOR,
    STATUS_NO_WINDOW,
    STATUS_OK,
    slice_windows,
)


def _msg(message_id, role, content):
    return {"message_id": message_id, "role": role, "content": content}


def _inj(event_id, anchor):
    return {"id": event_id, "payload": {"anchor": anchor}}


def test_basic_window_closes_at_next_injection():
    messages = [
        _msg("u1", "user", "q1"),
        _msg("a1", "assistant", "ans1"),
        _msg("u2", "user", "q2"),
        _msg("a2", "assistant", "ans2"),
    ]
    injections = [_inj(1, "u1"), _inj(2, "u2")]
    outcomes = slice_windows(messages, injections)
    assert [o.status for o in outcomes] == [STATUS_OK, STATUS_OK]
    assert outcomes[0].question == "q1"
    assert outcomes[0].answer == "ans1"
    assert outcomes[1].question == "q2"
    assert outcomes[1].answer == "ans2"


def test_last_window_extends_to_stream_end():
    messages = [
        _msg("u1", "user", "q1"),
        _msg("a1", "assistant", "ans1"),
        _msg("a1b", "assistant", "ans1 more"),
    ]
    outcomes = slice_windows(messages, [_inj(1, "u1")])
    assert outcomes[0].status == STATUS_OK
    assert outcomes[0].answer == "ans1\nans1 more"


def test_no_window_without_assistant_turn():
    messages = [_msg("u1", "user", "q1")]
    outcomes = slice_windows(messages, [_inj(1, "u1")])
    assert outcomes[0].status == STATUS_NO_WINDOW


def test_no_window_when_answer_empty_after_think_strip():
    messages = [
        _msg("u1", "user", "q1"),
        _msg("a1", "assistant", "<think>unclosed reasoning, truncated"),
    ]
    outcomes = slice_windows(messages, [_inj(1, "u1")])
    assert outcomes[0].status == STATUS_NO_WINDOW


def test_missing_anchor():
    messages = [_msg("u1", "user", "q1"), _msg("a1", "assistant", "ans")]
    outcomes = slice_windows(messages, [_inj(1, "gone")])
    assert outcomes[0].status == STATUS_MISSING_ANCHOR
    assert "gone" in outcomes[0].detail


def test_out_of_order_anchor_does_not_close_window():
    # A later injection anchored BEFORE the current one (clock skew / retry)
    # must not truncate the current window.
    messages = [
        _msg("u1", "user", "q1"),
        _msg("a1", "assistant", "ans1"),
        _msg("u2", "user", "q2"),
        _msg("a2", "assistant", "ans2"),
    ]
    injections = [_inj(1, "u2"), _inj(2, "u1")]
    outcomes = slice_windows(messages, injections)
    assert outcomes[0].question == "q2"
    assert outcomes[0].answer == "ans2"
