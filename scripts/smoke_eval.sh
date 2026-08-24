#!/bin/bash
# smoke_eval.sh — end-to-end smoke for `neatmem evaluate`.
#
# Runs --limit 1 (first conversation only) into an isolated, timestamped dir
# under tmp/smoke_eval/ (never touches runs/, never deletes prior smoke dirs).
# Asserts: full-stage artifacts, alias equivalence, alias conflict error,
# legacy EMBEDDING_* names ignored, and zero legacy names left in the tree.
#
# Prereqs: ./.env or exports with LLM + embedder keys (same as a real eval),
# and a qdrant binary (QDRANT_BIN env, or qdrant on PATH).
#
# Usage: bash scripts/smoke_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="tmp/smoke_eval/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
QBIN=()
if [ -n "${QDRANT_BIN:-}" ]; then QBIN=(--qdrant-bin "$QDRANT_BIN"); fi

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }
echo "== smoke output: $OUT"

# 1. Full pipeline, --limit 1: four artifact groups exist and parse.
echo "== [1/5] full run (ingest,search,judge, --limit 1)"
python -u -m neatmem.cli evaluate --output-dir "$OUT/full" --limit 1 \
  "${QBIN[@]}" > "$OUT/full.console.log" 2>&1 || {
  tail -30 "$OUT/full.console.log"; fail "full run non-zero exit"; }
python - "$OUT/full" <<'PY' || fail "artifact assertions"
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
m = json.loads((d / "manifest.json").read_text())
assert "score_mean" in m, "manifest missing score_mean"
r = json.loads((d / "results/run1/neatmem_results.json").read_text())
j = json.loads((d / "results/judge/judged_run1.json").read_text())
n_qa = sum(1 for v in r.values() for x in v if int(x.get("category", -1)) != 5)
# judged is keyed per judged QA task (llm_judge flattens and skips category
# 5), results per conversation
assert len(r) == 1 and len(j) == n_qa, f"convs={len(r)} judged={len(j)} qas={n_qa}"
s = (d / "results/score_run1.txt").read_text()
assert "Run score:" in s, "score file malformed"
print(f"    ok: score_mean={m['score_mean']:.4f}")
PY

# 2. Alias equivalence: --predict-only produces the ingest+search artifact
#    set (results present, no judge artifacts), and its dry-run stage line
#    matches --stages ingest,search.
echo "== [2/5] alias equivalence (--predict-only)"
python -u -m neatmem.cli evaluate --output-dir "$OUT/predict" --limit 1 \
  --predict-only "${QBIN[@]}" > "$OUT/predict.console.log" 2>&1 || {
  tail -30 "$OUT/predict.console.log"; fail "predict-only run non-zero exit"; }
[ -f "$OUT/predict/results/run1/neatmem_results.json" ] || fail "predict-only: results missing"
[ ! -e "$OUT/predict/results/judge/judged_run1.json" ] || fail "predict-only: judged file should not exist"
a=$(python -m neatmem.cli evaluate --predict-only --dry-run 2>/dev/null | grep "^Stages")
b=$(python -m neatmem.cli evaluate --stages ingest,search --dry-run 2>/dev/null | grep "^Stages")
[ "$a" = "$b" ] || fail "alias dry-run mismatch: '$a' vs '$b'"

# 3. Conflict: --stages search,judge + --predict-only must exit non-zero.
echo "== [3/5] alias conflict errors"
if python -m neatmem.cli evaluate --stages search,judge --predict-only \
     --dry-run > "$OUT/conflict.log" 2>&1; then
  fail "conflicting --stages/--predict-only did not error"
fi
grep -q "conflicts" "$OUT/conflict.log" || fail "conflict error message missing"

# 4. Rename: legacy EMBEDDING_PROVIDER is ignored (provider falls back to the
#    ./.env / default value, not the legacy process-env value).
echo "== [4/5] legacy EMBEDDING_* names ignored"
got=$(EMBEDDING_PROVIDER=openai python -c "from neatmem.config import EMBEDDER_PROVIDER as p; print(p)" 2>/dev/null)
[ "$got" = "siliconflow" ] || fail "legacy EMBEDDING_PROVIDER=openai leaked through (got $got)"

# 5. No legacy EMBEDDING_ identifiers left in shipped code/docs
#    (tests/test_embedder_config.py references them on purpose;
#    docs/internal-notes are dated historical records and are not rewritten).
echo "== [5/5] legacy name sweep"
if grep -rnI "EMBEDDING_" neatmem/ tests/ docs/ README.md .env.example requirements.txt \
     --exclude=test_embedder_config.py --exclude-dir=internal-notes \
     --exclude-dir=.ipynb_checkpoints --exclude-dir=__pycache__; then
  fail "legacy EMBEDDING_ references remain (see above)"
fi

echo "SMOKE PASS ($OUT)"
