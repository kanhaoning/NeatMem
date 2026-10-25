# Supported providers

NeatMem talks to OpenAI-compatible endpoints. Set `LLM_PROVIDER` and NeatMem
supplies the default base URL and the provider's thinking-control
parameters; `EMBEDDER_PROVIDER` does the same for embeddings.

```env
LLM_PROVIDER=minimax
LLM_API_KEY=...          # OPENAI_API_KEY also works
LLM_MODEL=MiniMax-M3
```

An explicit `OPENAI_BASE_URL` always **overrides** the provider preset —
remove stale values when switching providers, or the key and endpoint will
mismatch (typically a confusing 401).

## LLM providers

| `LLM_PROVIDER` | Aliases | Default endpoint | Thinking control | Notes |
|---|---|---|---|---|
| `minimax` | – | `https://api.minimaxi.com/v1` | native | M3 inlines `<think>` tags into content; NeatMem strips them. M2.5 is reasoning-only (no off switch exists) |
| `deepseek` | – | `https://api.deepseek.com` | native | |
| `dashscope` | `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | native | |
| `zhipu` | `glm`, `zai` | `https://open.bigmodel.cn/api/paas/v4` | native | |
| `moonshot` | `kimi` | `https://api.moonshot.cn/v1` | native | Rejects explicit `temperature` unless it matches the pinned per-thinking-state value, so NeatMem omits it |
| `volcengine` | `doubao`, `ark` | `https://ark.cn-beijing.volces.com/api/v3` | native | With thinking OFF the model writes its chain-of-thought into the visible answer; keep thinking ON for JSON-producing stages |
| `siliconflow` | – | `https://api.siliconflow.cn/v1` | native | Behavior depends on the hosted model: GLM/Kimi/Qwen3.5 controllable; MiniMax-M2.5 always thinks (but never leaks); DeepSeek-V3.2 spills CoT into content when off |
| `gemini` | – | `https://generativelanguage.googleapis.com/v1beta/openai/` | native | Controlled via `reasoning_effort`; thinking is never exposed in responses |
| `openrouter` | – | `https://openrouter.ai/api/v1` | via proxy | OpenRouter normalizes parameters; whether thinking can be disabled depends on the upstream model |
| `openai` | – | `https://api.openai.com/v1` | n/a | No off switch. NeatMem ports mem0's parameter filtering: exact reasoning models (o1/o3/gpt-5, …) drop sampling params and use `max_completion_tokens`; gpt-5.x variants keep `temperature` |

"native" means the endpoint accepts an explicit thinking on/off parameter.

Unknown `LLM_PROVIDER` values fail loudly at startup with the valid list.

## Embedding providers

| `EMBEDDER_PROVIDER` | Default endpoint | Max batch | Default model |
|---|---|---|---|
| `siliconflow` (default) | `https://api.siliconflow.cn/v1` | 100 | `BAAI/bge-m3` (1024 dims) |
| `openai` | `https://api.openai.com/v1` | 100 | set `EMBEDDER_MODEL` |
| `dashscope` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | **10** (hard API limit) | set `EMBEDDER_MODEL` |
| `xinference` | local server | – | `bge-m3` |

The batch limit matters: DashScope rejects batches above 10 with a 400, so
NeatMem chunks embedding requests per provider. `EMBEDDER_DIMS` is
auto-detected by a startup probe when unset.

## Cross-encoder providers

Used when `RERANK_MODE=cross_encoder`. Select with `CROSS_ENCODER_PROVIDER`;
`CROSS_ENCODER_MODEL` / `CROSS_ENCODER_BASE_URL` / `CROSS_ENCODER_API_KEY`
override the preset values.

| `CROSS_ENCODER_PROVIDER` | Default endpoint | Default model | Key env (fallback) |
|---|---|---|---|
| `siliconflow` (default) | `https://api.siliconflow.cn/v1` | `Qwen/Qwen3-Reranker-8B` | `SILICONFLOW_API_KEY` |
| `local` | in-process | `BAAI/bge-reranker-v2-m3` | – |

`local` loads the model in-process via sentence-transformers —
install with `pip install "neatmem[local-reranker]"` (`CROSS_ENCODER_DEVICE`
/ `CROSS_ENCODER_BATCH_SIZE` apply to `local` only).

Unknown `CROSS_ENCODER_PROVIDER` values fail loudly at startup with the
valid list.

## Per-stage thinking switches

Most stages have fixed defaults; two are configurable:

| Variable | Default | Stage |
|---|---|---|
| `DEDUP_THINKING` | `false` | Dedup decision call |
| `EDIT_THINKING` | `false` | Edit-mode patch generation |

All other stages send no thinking parameters (provider default applies —
unless a reasoning-effort knob is set, see below). Chinese providers
default to thinking **ON**, which costs extra tokens on every call — set
a provider explicitly and use the switches above where they apply.

## Per-stage reasoning effort

Some models can't turn thinking off at all (e.g. `MiniMax-M3.1-Flash`
rejects `thinking=disabled`) but accept a *reasoning effort* level
instead; some models accept both. Four stages expose an effort knob:

| Variable | Stage |
|---|---|
| `RERANK_REASONING_EFFORT` | LLM rerank |
| `DEDUP_REASONING_EFFORT` | Dedup decision calls |
| `EDIT_REASONING_EFFORT` | Edit-mode patch generation |
| `QUERY_REWRITE_REASONING_EFFORT` | Query rewrite |

The value is passed through **verbatim** — NeatMem does not validate it
against a whitelist, because valid values differ per provider *and per
model* (e.g. MiniMax accepts `low`/`medium`/`high`/`xhigh`/`max`;
OpenAI reasoning models accept `minimal`/`low`/`medium`/`high`; check
your provider's docs for your model). Instead, NeatMem verifies it for
you:

- **Launch probe**: when at least one effort knob is set, startup sends
  one minimal real request per stage that has a knob set. An invalid
  value fails the startup with the provider's error message verbatim
  (MiniMax's message lists the valid values). Transient errors (rate
  limit, 5xx, timeout) only log a warning — a provider outage must not
  block startup.
- **Loud failure at runtime**: when a knob is set and the provider still
  rejects a rerank/query-rewrite request as invalid (HTTP 400), the
  request fails visibly instead of being silently degraded (rerank
  falling back to score order, rewrite falling back to the original
  query) — a rejected effort value is an error you see, not a quiet
  quality regression. Write paths behave differently on purpose: if a
  dedup/edit request is rejected at runtime, NeatMem logs a warning and
  falls back to storing the memory as new rather than aborting your
  write — so for those stages the launch probe is the guard that catches
  a bad value.

Setting an effort knob implies thinking **ON** for that stage (effort
only makes sense with thinking active). For the two stages that also
have a thinking switch (dedup, edit), combining an effort knob with an
explicit `DEDUP_THINKING=false` / `EDIT_THINKING=false` is contradictory
and fails at startup. (Rerank and query rewrite have no thinking switch,
so no conflict is possible there.)

All knobs unset (the default) = zero behavior change: no extra startup
requests, identical API payloads.
