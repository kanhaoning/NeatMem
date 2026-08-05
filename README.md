# NeatMem

[![PyPI](https://img.shields.io/pypi/v/neatmem)](https://pypi.org/project/neatmem/)
[![Documentation](https://readthedocs.org/projects/neatmem/badge/?version=latest)](https://neatmem.readthedocs.io/en/latest/)

Lightweight local memory for agents, with cleaner deduplication, less memory pollution, and more relevant recall.

NeatMem is built for developers who want practical long-term memory without adopting a full Memory OS or hosted memory service. It focuses on keeping local agent memory clean: merging repeated facts, preventing AI suggestions, guesses, and tool noise from being saved as user facts, saving memories with enough context, and filtering irrelevant recalls.

> Status: v0.1-preview. NeatMem is usable for local development and mem0-compatible integrations, but APIs, packaging, and integrations may still change.

> **Benchmark**: 90.80% accuracy on LOCOMO, fully reproducible locally (3-run mean; MiniMax-M3 answer + judge, SiliconFlow bge-m3 embedding). See the [evaluation guide](https://github.com/kanhaoning/NeatMem/blob/main/neatmem/evaluation/README.md) for benchmark reproduction steps.

## Quick start

```bash
# 1. Install (the nlp extra is required by the default BM25 signal)
pip install "neatmem[nlp]"
python -m spacy download en_core_web_sm

# 2. Configure
curl -o .env https://raw.githubusercontent.com/kanhaoning/NeatMem/main/.env.example
# edit .env: OPENAI_API_KEY / OPENAI_BASE_URL / SILICONFLOW_API_KEY

# 3. Run
neatmem serve
```

The server listens on `http://localhost:8790`. Try it:

```python
from neatmem import MemoryClient

client = MemoryClient(host="http://localhost:8790")
client.add("My name is Alex", user_id="default_user")
results = client.search("What is my name?", filters={"user_id": "default_user"})
```

## Documentation

Full documentation: **[neatmem.readthedocs.io](https://neatmem.readthedocs.io/en/latest/)** (also readable as Markdown under [`docs/`](docs/))

- [Quick Start](docs/quickstart.md) — install options, minimal `.env`, server flags
- [Configuration](docs/configuration.md) — full environment variable reference
- [Custom Prompts](docs/custom-prompts.md) — replace any built-in prompt with your own
- [API Reference](docs/api.md) — mem0-compatible HTTP endpoints
- [Concepts](docs/concepts.md) — add/search flows and design notes
- Integrations: [OpenClaw](docs/integrations/openclaw.md) · [Hermes](docs/integrations/hermes.md)

## License

MIT License.

## Acknowledgements

NeatMem is inspired by the mem0 project and mem0-style memory API patterns, and is designed to interoperate with OpenClaw memory plugin flows. Upstream license notices should be preserved where applicable.

Some utility functions in `neatmem/utils/spacy/` (`spacy_models.py`, `entity_extraction.py`, `lemmatization.py`) are vendored from mem0 v2.0.0 (Apache-2.0); see file headers for modification notes.
