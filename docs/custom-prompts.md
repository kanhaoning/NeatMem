# Custom Prompts

Every core prompt can be replaced — either with a built-in variant id or with your own prompt file, `from_pretrained`-style. No code changes needed.

| Prompt | Env var / CLI flag | Built-in ids | Used when |
|---|---|---|---|
| Fact extraction | `EXTRACTION_PROMPT` / `--extraction-prompt` | - | always (write path) |
| Dedup decision | `DEDUP_PROMPT` / `--dedup-prompt` | `zh` (default), `en` | dedup enabled (listwise detector) |
| Merge rewrite | `REWRITE_PROMPT` / `--rewrite-prompt` | - | `DEDUP_RESOLVER=rewrite` |
| Patch edit | `EDIT_PROMPT` / `--edit-prompt` | - | `DEDUP_RESOLVER=edit` |
| Rerank | `LLM_RERANK_PROMPT` / `--rerank-prompt` | - | `RERANK_MODE=llm` (listwise or pointwise) |

Switch to the English dedup prompt:

```bash
neatmem serve --dedup-prompt en
```

Use your own prompt:

```bash
# 1. Export the packaged prompts as reference copies (all land as
#    *.example.txt, so re-exporting never overwrites your edits)
#    From a git clone:
mkdir -p custom_prompts
for f in neatmem/prompts/examples/*.txt; do
  b=$(basename "$f" .example.txt); b=${b%.txt}
  cp "$f" "custom_prompts/$b.example.txt"
done
#    From a pip install:
python - <<'EOF'
from importlib.resources import files
import shutil, os
os.makedirs("custom_prompts", exist_ok=True)
for f in files("neatmem.prompts").joinpath("examples").iterdir():
    name = f.name.removesuffix(".example.txt").removesuffix(".txt") + ".example.txt"
    shutil.copy(f, os.path.join("custom_prompts", name))
EOF

# 2. Copy the one you want to a working name, then edit the copy
#    (keep every {placeholder} intact, including the {{ }} escaping
#    in JSON examples; diff against the .example.txt original anytime)
cp custom_prompts/edit_en.example.txt custom_prompts/edit_en.txt

# 3. Point the server at your copy
neatmem serve --edit-prompt /absolute/path/to/custom_prompts/edit_en.txt
```

`edit_en.txt` and `rewrite_en.txt` are the actual default prompts the server loads; the exported `.example.txt` files for extraction/rerank/dedup mirror built-in code defaults (they converge in a later release).

Notes:

- Prompts are loaded once at startup; restart the server after editing a file.
- A value that is neither a known id nor an existing file, a missing file, or a missing `{placeholder}` fails at startup with a clear error. A `{placeholder}` the current code path does not supply also fails at startup — e.g. `{relation}` is only supplied on the pointwise dedup path (`DEDUP_DETECTOR=pointwise`).
- The two resolver prompts differ in who judges the relationship: the edit resolver decides supersede/append/conflict itself (listwise and pointwise share the same template), while the rewrite resolver consumes the detector's `{relation}` label directly.
- Prefer absolute paths in env vars; relative paths resolve against the server's working directory (CLI flags are anchored at the invocation directory).
