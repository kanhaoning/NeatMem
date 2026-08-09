# Quick Start

## 1. Install

```bash
pip install "neatmem[nlp]"
python -m spacy download en_core_web_sm
```

The `nlp` extra (spaCy + the English model) is required by the BM25 keyword search signal, which is enabled by default. For a minimal install without BM25, use `pip install neatmem` and set `ENABLE_BM25=false` in `.env`.

Optional:

- Local reranker model (alternative to LLM rerank):
  ```bash
  pip install "neatmem[local-reranker]"
  ```

Install from source (for development):

```bash
git clone https://github.com/kanhaoning/NeatMem.git
cd NeatMem
pip install -e ".[nlp]"
python -m spacy download en_core_web_sm
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

EMBEDDING_PROVIDER=siliconflow
SILICONFLOW_API_KEY=your-siliconflow-api-key
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
