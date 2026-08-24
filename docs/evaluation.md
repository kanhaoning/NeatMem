# Evaluation

NeatMem ships a one-command pipeline that reproduces the published LOCOMO
benchmark score locally. The LOCOMO-10 dataset is bundled with the package.

```bash
neatmem evaluate --dedup --dedup-resolver skip --output-dir runs/skip
```

This command (the `skip` strategy) is the reference reproduction of the
published score.

## How it works

Each run goes through the same pipeline:

1. **Ingest** — the dataset's conversations are written into NeatMem, which
   extracts and stores memories from them
2. **Search + answer** — for each question, memories are retrieved and an
   answer LLM responds based on them
3. **Judge** — a judge LLM scores each answer against the ground truth and
   prints per-category and overall accuracy

Runs are resumable: completed stages are skipped on re-run.

## Prerequisites

1. **NeatMem with the NLP extra** (the published score was measured with
   spaCy lemmatization for BM25):

   ```bash
   pip install "neatmem[nlp]" && python -m spacy download en_core_web_sm
   ```

2. **A qdrant server binary.** Each run uses an isolated Qdrant server
   process (the pip-installed embedded mode is not used here). Download a
   release binary for your platform from
   [github.com/qdrant/qdrant/releases](https://github.com/qdrant/qdrant/releases),
   extract it, and either pass `--qdrant-bin /path/to/qdrant`, set
   `QDRANT_BIN`, or put `qdrant` on your `PATH`.

3. **Provider keys** (LLM + embedding + judge), via exports or `./.env` —
   same configuration as [running the server](configuration.md):

   ```bash
   export OPENAI_API_KEY=your-key
   export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
   export LLM_MODEL=MiniMax-M3
   export EMBEDDER_PROVIDER=siliconflow
   export EMBEDDER_API_KEY=your-embedding-key
   ```

## What to expect

A full evaluation makes ~4,600 LLM calls (search + answer) plus ~1,540
judge calls. Wall time depends on your provider's rate limits — expect
hours, not minutes.

**Concurrency.** All stages default to 4 workers, safe for a single API
key. If you still hit rate-limit errors (HTTP 429), lower the per-stage
flags below. With several keys behind a local proxy (point
`OPENAI_BASE_URL` at it) or a high-quota provider, raise everything at once
with `--max-workers 16` (or more); per-stage flags override it.

Results, logs, scores, and a manifest (effective configuration with secrets
redacted, per-stage timings) land in `--output-dir` (default
`runs/<project-name>/`).

## Strategy variants

The dedup strategy is selected with command-line flags; provider and model
settings come from env as above. One run evaluates one strategy.

| Strategy | Command |
|---|---|
| skip | `neatmem evaluate --dedup --dedup-resolver skip --output-dir runs/skip` |
| edit | `neatmem evaluate --dedup --dedup-resolver edit --output-dir runs/edit` |
| replace | `neatmem evaluate --dedup --dedup-resolver replace --output-dir runs/replace` |
| rewrite | `neatmem evaluate --dedup --dedup-resolver rewrite --output-dir runs/rewrite` |
| pointwise-edit | `neatmem evaluate --dedup --dedup-detector pointwise --dedup-resolver edit --output-dir runs/pointwise-edit` |
| pointwise-rewrite | `neatmem evaluate --dedup --dedup-detector pointwise --dedup-resolver rewrite --output-dir runs/pointwise-rewrite` |
| off | `neatmem evaluate --no-dedup --output-dir runs/off` |

The first four differ in how a detected duplicate is resolved; the
`pointwise-*` variants detect duplicates per memory pair instead of in one
batched call (see `DEDUP_DETECTOR` in the
[configuration reference](configuration.md)).

Run the first four in one loop:

```bash
for r in skip edit replace rewrite; do
  neatmem evaluate --dedup --dedup-detector listwise --dedup-resolver $r --output-dir runs/$r
done
```

- Any `neatmem serve` flag also works here (e.g. `--rerank off --top-k 200`);
  flags are translated to env and applied to **all** stages, ingest included.
- Resuming is automatic: completed stages are skipped on re-run.
  `--stages ingest,search,judge` restricts which stages may run;
  `--reuse-db <path>` skips ingest and reuses an existing database.

## Options

Run identity and output:

| Option | Default | Description |
|---|---|---|
| `--project-name` | `default` | Run identifier; derives the output dir `runs/<name>` |
| `--output-dir` | `runs/<project-name>` | Where results, logs, and the manifest land (overrides the derived path) |
| `--runs` | `1` | Repeat search+judge this many times per run |
| `--limit` | all | Use only the first N conversations (smoke test) |
| `--dataset` | bundled LOCOMO-10 | Dataset path |
| `--env-file` | `./.env` | Bottom-layer env file |

Models (env fallback: `ANSWER_MODEL` / `JUDGE_MODEL`, then `LLM_MODEL`):

| Option | Default | Description |
|---|---|---|
| `--answerer-model` | env | Answer-generation model (env `ANSWER_MODEL`) |
| `--judge-model` | env | Judge model (env `JUDGE_MODEL`) |

Stages (resumption is automatic; these just restrict what may run):

| Option | Default | Description |
|---|---|---|
| `--stages` | all | Comma subset of `ingest,search,judge` |
| `--predict-only` | off | Shortcut for `--stages ingest,search` |
| `--evaluate-only` | off | Shortcut for `--stages judge` |
| `--reuse-db` | – | Skip ingest and reuse an existing database directory |
| `--force` | off | Re-run even where idempotency markers exist |

Concurrency (per-stage flag > `--max-workers` > built-in default):

| Option | Default | Description |
|---|---|---|
| `--max-workers` | `4` | Umbrella concurrency for all stages |
| `--ingest-workers` | `4` | Ingest concurrency |
| `--search-workers` | `4` | Search+answer concurrency |
| `--judge-workers` | `4` | Judge concurrency |

Other:

| Option | Default | Description |
|---|---|---|
| `--top-k` | `20` | Retrieval depth (env `TOP_K`) |
| `--batch-size` | `10` | Messages per extraction batch at ingest (env `BATCH_SIZE`) |
| `--qdrant-bin` | `PATH` lookup | qdrant server binary path |
| `--dry-run` | off | Print preflight and merged env without executing |

If `--stages` and the boolean shortcuts are given together, they must agree
(`--stages ingest,search --predict-only` is fine, `--stages search,judge
--predict-only` is an error).

### For mem0 benchmark users

Flag names follow mem0's memory-benchmarks CLI where the semantics match
(`--project-name`, `--answerer-model`, `--judge-model`, `--max-workers`,
`--predict-only`, `--evaluate-only`). Two differences to know:

- **Answer generation lives inside the search stage**, so `--predict-only`
  still calls the answer model. `--evaluate-only` re-judges existing
  answers; it does not re-generate them.
- There is no `--resume`: every run resumes automatically from idempotency
  markers. There is also no `--provider` flag; providers are configured via
  env (`LLM_PROVIDER`, `EMBEDDER_PROVIDER`), see
  [providers](providers.md).

## Configuration

When the same variable is set in several places, the later source in this
list wins: `./.env` or `--env-file` < process env < serve/evaluate flags <
values fixed by the pipeline itself (ports, per-run database paths). The
`.env` file is optional convenience; exports alone are fully supported.

The strategy variables (`DEDUP_ENABLED`, `DEDUP_RESOLVER`, `DEDUP_DETECTOR`,
`ENABLE_BM25`, `ENABLE_ENTITY`, `RERANK_MODE`, `*_THINKING`) are documented
in the [configuration reference](configuration.md).

## Manual steps

`neatmem evaluate` automates exactly these steps — useful for debugging or
custom pipelines:

```bash
# 1. Start the server
python -m neatmem.main

# 2. Ingest the dataset
python -m neatmem.evaluation.run_experiments --method add --dataset neatmem/evaluation/dataset/locomo10.json

# 3. Search + answer
python -m neatmem.evaluation.run_experiments \
  --method search \
  --dataset neatmem/evaluation/dataset/locomo10.json \
  --output-folder results/ \
  --top-k 20 \
  --workers 8

# 4. Judge
python -m neatmem.evaluation.metrics.llm_judge \
  --input_file results/neatmem_results.json \
  --output_file results/judged.json \
  --workers 8
```

The judge prints per-category and overall accuracy to stdout:

```text
Final summary:
Total: X/1540 = 0.XXXX
  Category 1: ...
  ...
```

## Results

| Config | 3-run mean | Date |
|---|---|---|
| `DEDUP_RESOLVER=skip` | 0.9080 | 2026-08 |

Model stack: MiniMax-M3 (answer + judge), SiliconFlow bge-m3 (embedding).
