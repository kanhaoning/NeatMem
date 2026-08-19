# NeatMem Evaluation

LOCOMO benchmark evaluation for NeatMem.

## Prerequisites

```bash
pip install -e .
```

The evaluation config below sets `ENABLE_BM25=true`. The benchmark score was measured with the full `[nlp]` configuration (spaCy lemmatization for BM25):

```bash
pip install -e ".[nlp]" && python -m spacy download en_core_web_sm
```

Configure environment (`.env` or shell):

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=MiniMax-M3
EMBEDDING_PROVIDER=siliconflow
SILICONFLOW_API_KEY=your-siliconflow-key
DEDUP_RESOLVER=skip
ENABLE_BM25=true
ENABLE_ENTITY=false
LLM_RERANK=true
```

> **Rate limits**: a full evaluation makes ~4600 LLM calls (search+answer) plus ~1540 judge calls. A single API key may hit 429 rate limits; if so, lower `--workers`, or run several keys behind a local load-balancing proxy and point `OPENAI_BASE_URL` at it.

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

| Variable | Default | Description |
|---|---|---|
| `DEDUP_ENABLED` | `true` | Enable dedup on write |
| `DEDUP_RESOLVER` | `skip` | Duplicate resolution: `skip`, `replace`, `rewrite`, `edit` |
| `DEDUP_DETECTOR` | `listwise` | Duplicate detection: `listwise`, `pointwise` |
| `ENABLE_BM25` | `true` | BM25 sparse search signal |
| `ENABLE_ENTITY` | `false` | Entity extraction and boosting |
| `LLM_RERANK` | `true` | LLM listwise rerank |
| `DEDUP_THINKING` | `false` | LLM thinking for dedup |
| `EDIT_THINKING` | `false` | LLM thinking for edit mode |

## Results

| Config | 3-run mean | Date |
|---|---|---|
| `DEDUP_RESOLVER=skip` | 0.9080 | 2026-08 |

Model stack: MiniMax-M3 (answer + judge), SiliconFlow bge-m3 (embedding).
