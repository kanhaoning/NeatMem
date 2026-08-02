"""NeatMem command-line interface.

Flags are syntactic sugar over env injection: parse args -> load .env ->
inject into os.environ -> uvicorn.run("neatmem.main:app"). The string-form
app import delays neatmem.main (and config.py) until after injection, so
import-time env reads see the flag values.

Priority: CLI flag > existing environment variable > .env file
(load_dotenv does not override existing env; flags are injected after).
"""

import argparse
import os
import sys
from urllib.parse import urlparse

from neatmem import __version__

# NEATMEM_* vars the CLI itself consumes; anything else with this prefix is
# likely a typo and worth a startup warning (unprefixed vars cannot be
# distinguished from system env, so they are not checked).
_KNOWN_NEATMEM_ENV = {"NEATMEM_HOST", "NEATMEM_PORT", "NEATMEM_URL", "NEATMEM_API_KEY"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neatmem",
        description="NeatMem — a local mem0-compatible memory server for AI agents",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start the NeatMem server")
    serve.add_argument("--host", help="Bind host (env NEATMEM_HOST, default 0.0.0.0)")
    serve.add_argument("--port", type=int, help="Bind port (env NEATMEM_PORT, default 8790)")
    serve.add_argument("--env-file", help="Path to .env file (default: .env in current directory)")

    serve.add_argument("--llm-model", help="LLM model name (env LLM_MODEL)")
    serve.add_argument("--llm-api-key", help="LLM API key (env OPENAI_API_KEY)")
    serve.add_argument("--llm-base-url", help="LLM base URL (env OPENAI_BASE_URL)")

    serve.add_argument("--embedding-model", help="Embedding model name (env EMBEDDING_MODEL)")
    serve.add_argument("--embedding-base-url", help="Embedding base URL (env EMBEDDING_BASE_URL)")
    serve.add_argument("--embedding-api-key", help="Embedding API key (env SILICONFLOW_API_KEY)")

    vdb = serve.add_mutually_exclusive_group()
    vdb.add_argument("--vector-db-path", help="Embedded vector DB directory (env QDRANT_PATH)")
    vdb.add_argument("--vector-db-url", help="Vector DB server URL, e.g. http://localhost:6333 (env QDRANT_HOST/PORT)")

    serve.add_argument("--history-db-path", help="Message history sqlite path "
                       "(env HISTORY_DB_PATH, default <vector-db-path>/history.db)")

    serve.add_argument("--enable-bm25", action=argparse.BooleanOptionalAction, default=None,
                       help="BM25 sparse search signal (env ENABLE_BM25, default true)")
    serve.add_argument("--enable-entity", action=argparse.BooleanOptionalAction, default=None,
                       help="Entity extraction and boosting (env ENABLE_ENTITY, default false)")
    serve.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                       help="LLM listwise rerank (env LLM_RERANK, default true)")
    serve.add_argument("--enable-graph", action=argparse.BooleanOptionalAction, default=None,
                       help="Graph memory store, opt-in (env ENABLE_GRAPH, default false)")

    serve.add_argument("--dedup-mode", choices=["off", "skip", "replace", "rewrite", "edit"],
                       help="Dedup behavior (env DEDUP_MODE, default skip)")
    serve.add_argument("--dedup-thinking", action=argparse.BooleanOptionalAction, default=None,
                       help="LLM thinking for dedup (env DEDUP_THINKING, default false)")

    serve.add_argument("--extract-last-k-messages", type=int,
                       help="Extraction context window (env EXTRACT_LAST_K_MESSAGES, default 10)")

    serve.add_argument("--embedding-dims", type=int,
                       help="Embedding dimensions (env EMBEDDING_DIMS, default: auto-detect from API probe)")

    serve.add_argument("--extraction-prompt", help="Custom extraction prompt: built-in id or file path (env EXTRACTION_PROMPT)")
    serve.add_argument("--dedup-prompt", help="Custom dedup prompt: built-in id (zh/en) or file path (env DEDUP_PROMPT)")
    serve.add_argument("--rewrite-prompt", help="Custom merge/rewrite prompt: built-in id or file path (env REWRITE_PROMPT)")
    serve.add_argument("--edit-prompt", help="Custom patch/edit prompt: built-in id or file path (env EDIT_PROMPT)")
    serve.add_argument("--rerank-prompt", help="Custom rerank prompt: built-in id or file path (env RERANK_PROMPT)")
    return parser


def _warn_unknown_neatmem_env() -> None:
    for key in sorted(os.environ):
        if key.startswith("NEATMEM_") and key not in _KNOWN_NEATMEM_ENV:
            print(f"neatmem: warning: unrecognized env var {key} (typo?)", file=sys.stderr)


def _inject(args: argparse.Namespace) -> None:
    str_flags = {
        "NEATMEM_HOST": args.host,
        "LLM_MODEL": args.llm_model,
        "OPENAI_API_KEY": args.llm_api_key,
        "OPENAI_BASE_URL": args.llm_base_url,
        "EMBEDDING_MODEL": args.embedding_model,
        "EMBEDDING_BASE_URL": args.embedding_base_url,
        "SILICONFLOW_API_KEY": args.embedding_api_key,
        "HISTORY_DB_PATH": args.history_db_path,
        "DEDUP_MODE": args.dedup_mode,
        "EXTRACTION_PROMPT": args.extraction_prompt,
        "DEDUP_PROMPT": args.dedup_prompt,
        "REWRITE_PROMPT": args.rewrite_prompt,
        "EDIT_PROMPT": args.edit_prompt,
        "RERANK_PROMPT": args.rerank_prompt,
    }
    for env_key, value in str_flags.items():
        if value is not None:
            os.environ[env_key] = value
    if args.port is not None:
        os.environ["NEATMEM_PORT"] = str(args.port)
    if args.extract_last_k_messages is not None:
        os.environ["EXTRACT_LAST_K_MESSAGES"] = str(args.extract_last_k_messages)
    if args.embedding_dims is not None:
        os.environ["EMBEDDING_DIMS"] = str(args.embedding_dims)

    bool_flags = {
        "ENABLE_BM25": args.enable_bm25,
        "ENABLE_ENTITY": args.enable_entity,
        "LLM_RERANK": args.rerank,
        "ENABLE_GRAPH": args.enable_graph,
        "DEDUP_THINKING": args.dedup_thinking,
    }
    for env_key, value in bool_flags.items():
        if value is not None:
            os.environ[env_key] = "true" if value else "false"

    if args.vector_db_path is not None:
        os.environ["QDRANT_PATH"] = args.vector_db_path
        # Explicit path means embedded mode; clear any inherited host so it
        # does not silently win in config.py (host non-empty -> server mode).
        os.environ["QDRANT_HOST"] = ""
    if args.vector_db_url is not None:
        url = args.vector_db_url if "://" in args.vector_db_url else f"http://{args.vector_db_url}"
        parsed = urlparse(url)
        if not parsed.hostname:
            raise SystemExit(f"neatmem: error: invalid --vector-db-url {args.vector_db_url!r}")
        os.environ["QDRANT_HOST"] = parsed.hostname
        os.environ["QDRANT_PORT"] = str(parsed.port or 6333)


def _serve(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv(args.env_file or ".env")  # existing env wins over .env
    _inject(args)
    _warn_unknown_neatmem_env()

    import uvicorn

    host = os.environ.get("NEATMEM_HOST", "0.0.0.0")
    port = int(os.environ.get("NEATMEM_PORT", "8790"))
    uvicorn.run("neatmem.main:app", host=host, port=port)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        _serve(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
