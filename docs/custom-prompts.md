# Custom Prompts

Every core prompt can be replaced — either with a built-in variant id or with your own prompt file, `from_pretrained`-style. No code changes needed.

| Prompt | Env var / CLI flag | Built-in ids | Used when |
|---|---|---|---|
| Fact extraction | `EXTRACTION_PROMPT` / `--extraction-prompt` | - | always (write path) |
| Dedup decision | `DEDUP_PROMPT` / `--dedup-prompt` | `zh` (default), `en` | dedup enabled (listwise detector) |
| Merge rewrite | `REWRITE_PROMPT` / `--rewrite-prompt` | - | `DEDUP_RESOLVER=rewrite` |
| Patch edit | `EDIT_PROMPT` / `--edit-prompt` | - | `DEDUP_RESOLVER=edit` |
| Rerank | `RERANK_PROMPT` / `--rerank-prompt` | - | LLM listwise rerank |

Switch to the English dedup prompt:

```bash
neatmem serve --dedup-prompt en
```

Use your own prompt:

```bash
# 1. Export the example templates (they are the exact built-in defaults)
#    From a git clone:
mkdir -p my_prompts && cp neatmem/prompts/examples/*.txt my_prompts/
#    From a pip install:
python - <<'EOF'
from importlib.resources import files
import shutil, os
os.makedirs("my_prompts", exist_ok=True)
for f in files("neatmem.prompts").joinpath("examples").iterdir():
    shutil.copy(f, "my_prompts/")
EOF

# 2. Edit the one you want (keep every {placeholder} intact,
#    including the {{ }} escaping in JSON examples)

# 3. Point the server at it
neatmem serve --dedup-prompt /absolute/path/to/my_prompts/dedup_zh.example.txt
```

Notes:

- Prompts are loaded once at startup; restart the server after editing a file.
- A value that is neither a known id nor an existing file, a missing file, or a missing `{placeholder}` fails at startup with a clear error.
- Prefer absolute paths; relative paths resolve against the server's working directory.
