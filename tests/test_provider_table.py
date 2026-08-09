"""Provider table tests — land at tests/test_provider_table.py on merge.

Run against the experiment copy:
  PYTHONPATH=neatmem_pkg python -m pytest test_provider_table.py -q
(or plain: python test_provider_table.py)

Covers: legacy path byte-identity (the compatibility bottom line),
per-provider thinking shapes, constraint rewriting, alias resolution,
and loud failure on unknown providers. No network access.
"""

import pytest

from neatmem.utils.llm_client import (
    PROVIDER_TABLE,
    apply_provider_constraints,
    build_thinking_extra,
    is_reasoning_model,
    normalize_provider,
    provider_default_base_url,
    uses_max_completion_tokens,
)

# --- Legacy path (provider=None) must stay byte-identical to pre-change ---

def test_legacy_minimax_dual_field_unchanged():
    assert build_thinking_extra("MiniMax-M3", True) == {
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking": {"type": "adaptive"},
    }
    assert build_thinking_extra("MiniMax-M3", False) == {
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }


def test_legacy_qwen_deepseek_unknown_unchanged():
    assert build_thinking_extra("qwen3.7-max", False) == {
        "chat_template_kwargs": {"enable_thinking": False}}
    assert build_thinking_extra("deepseek-v4-pro", True) == {
        "thinking": {"type": "enabled"}}
    assert build_thinking_extra("some-unknown-model", True) == {}


def test_legacy_constraints_noop_without_provider():
    params = {"temperature": 0.0, "max_tokens": 2000}
    assert apply_provider_constraints(params, "kimi-k2.6", None) == params


# --- Explicit provider shapes (each backed by smoke evidence) ---

@pytest.mark.parametrize("provider", ["deepseek", "zhipu", "moonshot",
                                      "volcengine", "siliconflow"])
def test_thinking_type_shape(provider):
    assert build_thinking_extra("any-model", False, provider) == {
        "thinking": {"type": "disabled"}}
    assert build_thinking_extra("any-model", True, provider) == {
        "thinking": {"type": "enabled"}}


def test_dashscope_ctk_shape():
    assert build_thinking_extra("m", False, "dashscope") == {
        "chat_template_kwargs": {"enable_thinking": False}}


def test_minimax_explicit_provider_matches_legacy():
    assert build_thinking_extra("m", True, "minimax") == \
        build_thinking_extra("MiniMax-M3", True)


def test_passthrough_and_openai_send_no_extra_body():
    assert build_thinking_extra("m", False, "openrouter") == {}
    assert build_thinking_extra("m", False, "openai") == {}


# --- Aliases and validation ---

def test_aliases_resolve_to_canonical():
    assert normalize_provider("qwen") == "dashscope"
    assert normalize_provider("GLM") == "zhipu"
    assert normalize_provider("kimi") == "moonshot"
    assert normalize_provider("doubao") == "volcengine"
    assert normalize_provider(None) is None
    assert normalize_provider("") is None


def test_unknown_provider_fails_loudly():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        normalize_provider("depseek")


def test_every_provider_has_base_url():
    for name, row in PROVIDER_TABLE.items():
        assert row["base_url"].startswith("https://"), name
        assert provider_default_base_url(name) == row["base_url"]


# --- Constraints ---

def test_moonshot_drops_temperature():
    p = apply_provider_constraints({"temperature": 0.0, "max_tokens": 2000},
                                   "kimi-k2.6", "moonshot")
    assert "temperature" not in p
    assert p["max_tokens"] == 2000


def test_openai_reasoning_filter_matches_mem0_heuristic():
    # exact reasoning names: drop sampling params + rename max_tokens
    p = apply_provider_constraints(
        {"temperature": 0.1, "top_p": 0.1, "max_tokens": 2000}, "gpt-5", "openai")
    assert "temperature" not in p and "top_p" not in p
    assert p["max_completion_tokens"] == 2000 and "max_tokens" not in p
    # gpt-5.x variants keep temperature (mem0: gpt-5.4-mini supports it)
    p = apply_provider_constraints({"temperature": 0.1, "max_tokens": 2000},
                                   "gpt-5-nano", "openai")
    assert p == {"temperature": 0.1, "max_completion_tokens": 2000}
    # non-reasoning untouched
    p = apply_provider_constraints({"temperature": 0.1, "max_tokens": 2000},
                                   "gpt-4o", "openai")
    assert p == {"temperature": 0.1, "max_tokens": 2000}
    assert is_reasoning_model("o3-mini-2025-01-31")
    assert is_reasoning_model("openai/o3")  # provider prefix stripped
    assert not is_reasoning_model("gpt-5.4-mini")
    assert uses_max_completion_tokens("gpt-5.4-mini")


def test_gemini_reasoning_effort_toggle():
    p = apply_provider_constraints({"temperature": 0.0}, "gemini-2.5-flash",
                                   "gemini", thinking_enable=False)
    assert p["reasoning_effort"] == "none"
    p = apply_provider_constraints({"temperature": 0.0}, "gemini-2.5-flash",
                                   "gemini", thinking_enable=True)
    assert "reasoning_effort" not in p


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
