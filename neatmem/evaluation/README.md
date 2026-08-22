# NeatMem Evaluation

LOCOMO benchmark evaluation for NeatMem.

## Prerequisites

```bash
pip install "neatmem[nlp]" && python -m spacy download en_core_web_sm
```

(`[nlp]` gives spaCy lemmatization for BM25; the benchmark score was measured with it.)

Export your provider settings (or put them in `./.env` — see Configuration):

```bash
export OPENAI_API_KEY=your-key
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export LLM_MODEL=MiniMax-M3
export EMBEDDING_PROVIDER=siliconflow
export SILICONFLOW_API_KEY=your-siliconflow-key
```

The qdrant server binary is also required: pass `--qdrant-bin`, set `QDRANT_BIN`, or put `qdrant` on `PATH`. Download a release binary from https://github.com/qdrant/qdrant/releases (pick your platform, extract, pass the binary path).

> **Rate limits**: a full evaluation makes ~4600 LLM calls (search+answer) plus ~1540 judge calls. A single API key may hit 429 rate limits; if so, lower `--workers`, or run several keys behind a local load-balancing proxy and point `OPENAI_BASE_URL` at it.

## Strategy arms: `neatmem evaluate`

One command runs the whole pipeline (qdrant → ingest → search → judge → score), resumable per stage. One run = one strategy: env (exports or `./.env`) fixes the infrastructure, serve flags pick the strategy:

```bash
neatmem evaluate --dedup --dedup-detector listwise --dedup-resolver skip --output-dir runs/skip
```

This (the `skip` arm) is the reference reproduction command for the published score below.

| Arm | Command |
|---|---|
| skip | `neatmem evaluate --dedup --dedup-resolver skip --output-dir runs/skip` |
| edit | `neatmem evaluate --dedup --dedup-resolver edit --output-dir runs/edit` |
| replace | `neatmem evaluate --dedup --dedup-resolver replace --output-dir runs/replace` |
| rewrite | `neatmem evaluate --dedup --dedup-resolver rewrite --output-dir runs/rewrite` |
| pointwise-edit | `neatmem evaluate --dedup --dedup-detector pointwise --dedup-resolver edit --output-dir runs/pointwise-edit` |
| pointwise-rewrite | `neatmem evaluate --dedup --dedup-detector pointwise --dedup-resolver rewrite --output-dir runs/pointwise-rewrite` |
| off | `neatmem evaluate --no-dedup --output-dir runs/off` |

All listwise arms in one loop:

```bash
for r in skip edit replace rewrite; do
  neatmem evaluate --dedup --dedup-detector listwise --dedup-resolver $r --output-dir runs/$r
done
```

- Any `neatmem serve` flag also works here (e.g. `--rerank off --top-k 200`); flags are translated to env and applied to **all** stages, ingest included.
- Results, logs, and a manifest (effective env with secrets redacted, scores, per-stage timings) land in `--output-dir` (default `runs/default/`).

The manual steps below are exactly what `neatmem evaluate` automates.

## Steps

### 1. Start NeatMem server

```bash
python -m neatmem.main
```

Server listens on `http://localhost:8790`.

### 2. Ingest LOCOMO dataset

```bash
python -m neatmem.evaluation.run_experiments --method add --dataset neatmem/evaluation/dataset/locomo10.json
```

### 3. Search + Answer

```bash
python -m neatmem.evaluation.run_experiments \
  --method search \
  --dataset neatmem/evaluation/dataset/locomo10.json \
  --output-folder results/ \
  --top-k 20 \
  --workers 8
```

Output: `results/neatmem_results.json`

### 4. Judge

```bash
python -m neatmem.evaluation.metrics.llm_judge \
  --input_file results/neatmem_results.json \
  --output_file results/judged.json \
  --workers 8
```

### 5. Score

Judge outputs per-category and overall accuracy to stdout. Expected output:

```
Final summary:
Total: X/1540 = 0.XXXX
  Category 1: ...
  Category 2: ...
  Category 3: ...
  Category 4: ...
```

## Configuration

Env layering (later wins): `./.env` or `--env-file` < process env < serve/eval flags < orchestrator-forced items (ports, per-run db paths). The `.env` file is optional convenience; exports alone are fully supported.

| Variable | Default | Description |
|---|---|---|
| `DEDUP_ENABLED` | `true` | Enable dedup on write |
| `DEDUP_RESOLVER` | `skip` | Duplicate resolution: `skip`, `replace`, `rewrite`, `edit` |
| `DEDUP_DETECTOR` | `listwise` | Duplicate detection: `listwise`, `pointwise` |
| `ENABLE_BM25` | `true` | BM25 sparse search signal |
| `ENABLE_ENTITY` | `false` | Entity extraction and boosting |
| `RERANK_MODE` | `llm` | Rerank engine: `llm`, `cross_encoder`, `off` |
| `DEDUP_THINKING` | `false` | LLM thinking for dedup |
| `EDIT_THINKING` | `false` | LLM thinking for edit mode |

## Results

| Config | 3-run mean | Date |
|---|---|---|
| `DEDUP_RESOLVER=skip` | 0.9080 | 2026-08 |

Model stack: MiniMax-M3 (answer + judge), SiliconFlow bge-m3 (embedding).
