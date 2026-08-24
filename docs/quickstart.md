# Quick Start

## 1. Install

```bash
pip install neatmem
```

Optional:

- Enhanced BM25 keyword matching (lemmatization, so searching "memory" also matches "memories"):
  ```bash
  pip install "neatmem[nlp]"
  python -m spacy download en_core_web_sm
  ```
- Local reranker model (alternative to LLM rerank):
  ```bash
  pip install "neatmem[local-reranker]"
  ```

Install from source (for development):

```bash
git clone https://github.com/kanhaoning/NeatMem.git
cd NeatMem
pip install -e .
```

## 2. Configure environment variables

Fetch the full `.env` template (includes commented optional settings):

```bash
curl -o .env https://raw.githubusercontent.com/kanhaoning/NeatMem/main/.env.example
```

Or create a `.env` file manually. Minimum configuration (MiniMax example; see [configuration](configuration.md) for other providers):

```env
LLM_PROVIDER=minimax
LLM_API_KEY=your-minimax-api-key
LLM_MODEL=MiniMax-M3

EMBEDDER_PROVIDER=siliconflow
EMBEDDER_API_KEY=your-siliconflow-api-key
```

## 3. Start the server

```bash
neatmem serve
```

The server listens on:

```text
http://localhost:8790
```

To use a different port:

```bash
neatmem serve --port 9000
```

View all options:

```bash
neatmem serve --help
```

CLI flags override `.env` environment variables; see the [Configuration](configuration.md) table for the full list.

Alternatively, start directly with Python:

```bash
python -m neatmem.main
```

Check health:

```bash
curl http://localhost:8790/health
```

Expected response:

```json
{"status":"healthy","timestamp":"..."}
```

Next: call the API with curl or the [Python client](client.md), or reproduce
the LOCOMO benchmark with [one command](evaluation.md).
