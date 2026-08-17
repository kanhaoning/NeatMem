# Supported providers

NeatMem talks to OpenAI-compatible endpoints. Set `LLM_PROVIDER` and NeatMem
supplies the default base URL and the provider's verified thinking-control
parameters; `EMBEDDING_PROVIDER` does the same for embeddings.

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
| `minimax` | – | `https://api.minimaxi.com/v1` | ✅ verified (native) | M3 inlines `<think>` tags into content; NeatMem strips them. M2.5 is reasoning-only (no off switch exists) |
| `deepseek` | – | `https://api.deepseek.com` | ✅ verified (native) | |
| `dashscope` | `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | ✅ verified (native) | |
| `zhipu` | `glm` | `https://open.bigmodel.cn/api/paas/v4` | ✅ verified (native) | |
| `moonshot` | `kimi` | `https://api.moonshot.cn/v1` | ✅ verified (native) | Rejects explicit `temperature` unless it matches the pinned per-thinking-state value, so NeatMem omits it |
| `volcengine` | `doubao`, `ark` | `https://ark.cn-beijing.volces.com/api/v3` | ✅ verified (native) | With thinking OFF the model writes its chain-of-thought into the visible answer; keep thinking ON for JSON-producing stages |
| `siliconflow` | – | `https://api.siliconflow.cn/v1` | ✅ verified (native) | Behavior depends on the hosted model: GLM/Kimi/Qwen3.5 controllable; MiniMax-M2.5 always thinks (but never leaks); DeepSeek-V3.2 spills CoT into content when off |
| `gemini` | – | `https://generativelanguage.googleapis.com/v1beta/openai/` | ✅ verified (native) | Controlled via `reasoning_effort`; thinking is never exposed in responses |
| `openrouter` | – | `https://openrouter.ai/api/v1` | ✅ verified (proxy) | OpenRouter normalizes parameters; whether thinking can be disabled depends on the upstream model |
| `openai` | – | `https://api.openai.com/v1` | mem0 evidence | No off switch. NeatMem ports mem0's parameter filtering: exact reasoning models (o1/o3/gpt-5, …) drop sampling params and use `max_completion_tokens`; gpt-5.x variants keep `temperature` |

"Thinking control ✅" means: a reasoning-prone prompt was run with thinking
explicitly ON and OFF against the merged NeatMem code, and the response
gained/lost its reasoning accordingly (reasoning-token counts in the
hundreds vs ~0). Raw responses are archived in the project.

Unknown `LLM_PROVIDER` values fail loudly at startup with the valid list.

## Embedding providers

| `EMBEDDING_PROVIDER` | Default endpoint | Max batch | Default model |
|---|---|---|---|
| `siliconflow` (default) | `https://api.siliconflow.cn/v1` | 100 | `BAAI/bge-m3` (1024 dims) |
| `openai` | `https://api.openai.com/v1` | 100 | set `EMBEDDING_MODEL` |
| `dashscope` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | **10** (hard API limit, verified) | set `EMBEDDING_MODEL` |
| `xinference` | local server | – | `bge-m3` |

The batch limit matters: DashScope rejects batches above 10 with a 400, so
NeatMem chunks embedding requests per provider. `EMBEDDING_DIMS` is
auto-detected by a startup probe when unset.

## Per-stage thinking switches

Most stages have fixed, verified defaults; two are configurable:

| Variable | Default | Stage |
|---|---|---|
| `DEDUP_THINKING` | `false` | Dedup decision call |
| `EDIT_THINKING` | `false` | Edit-mode patch generation |

All other stages send no thinking parameters (provider default applies).
Note that Chinese providers default to thinking **ON**, which costs extra
tokens on every call — set a provider explicitly and use the switches above
where they apply.
