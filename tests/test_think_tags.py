"""Unit tests for neatmem.utils.think_tags.strip_think_tags."""

from neatmem.utils.think_tags import strip_think_tags


def test_plain_text_unchanged():
    assert strip_think_tags("hello world") == "hello world"


def test_empty_and_none():
    assert strip_think_tags("") == ""
    assert strip_think_tags(None) == ""


def test_closed_pair():
    assert strip_think_tags("<think>inner monologue</think>visible answer") == "visible answer"


def test_closed_pair_with_attributes():
    assert strip_think_tags('<think type="x">inner</think>answer') == "answer"


def test_closed_pair_thinking_variant():
    assert strip_think_tags("<think>inner</thinking>answer") == "answer"


def test_multiple_closed_pairs():
    text = "<think>a</think>first<think>b</think>second"
    assert strip_think_tags(text) == "firstsecond"


def test_multiline_think_block():
    text = "<think>\nline1\nline2\n</think>\nanswer"
    assert strip_think_tags(text) == "answer"


def test_unclosed_stripped_to_end():
    assert strip_think_tags("visible<think>truncated forever") == "visible"


def test_unclosed_whole_message():
    assert strip_think_tags("<think>8000 chars of reasoning, never closed") == ""


def test_orphan_close_keeps_tail():
    assert strip_think_tags("lost head continuation</think>visible") == "visible"


def test_orphan_close_thinking_variant():
    assert strip_think_tags("lost head</thinking>visible") == "visible"


def test_closed_then_unclosed():
    text = "<think>a</think>kept<think>b truncated"
    assert strip_think_tags(text) == "kept"


def test_all_think_message_yields_empty():
    assert strip_think_tags("<think>only reasoning</think>") == ""


def test_user_style_literal_discussion_untouched_when_no_tags():
    # Plain prose mentioning think without actual tags stays as-is.
    assert strip_think_tags("how do I use think blocks?") == "how do I use think blocks?"
