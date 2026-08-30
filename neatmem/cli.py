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


def add_serve_arguments(serve: argparse.ArgumentParser) -> None:
    """All `serve` flags. Shared by the serve subcommand and `neatmem evaluate`,
    which reuses this definition to validate/translate passthrough flags."""
    serve.add_argument("--host", help="Bind host (env NEATMEM_HOST, default 0.0.0.0)")
    serve.add_argument("--port", type=int, help="Bind port (env NEATMEM_PORT, default 8790)")
    serve.add_argument("--env-file", help="Path to .env file (default: .env in current directory)")

    serve.add_argument("--llm-model", help="LLM model name (env LLM_MODEL)")
    serve.add_argument("--llm-api-key", help="LLM API key (env LLM_API_KEY, fallback OPENAI_API_KEY)")
    serve.add_argument("--llm-base-url", help="LLM base URL (env OPENAI_BASE_URL)")
    serve.add_argument("--llm-provider",
                       help="LLM provider for verified thinking/param handling "
                            "(env LLM_PROVIDER): deepseek|dashscope|zhipu|moonshot|volcengine|"
                            "minimax|siliconflow|openai|gemini|openrouter "
                            "(aliases: qwen, glm, kimi, doubao)")

    serve.add_argument("--embedder-model", help="Embedding model name (env EMBEDDER_MODEL)")
    serve.add_argument("--embedder-base-url", help="Embedding base URL (env EMBEDDER_BASE_URL)")
    serve.add_argument("--embedder-api-key", help="Embedding API key (env EMBEDDER_API_KEY, fallback SILICONFLOW_API_KEY)")
    serve.add_argument("--embedder-provider",
                       help="Embedding provider (env EMBEDDER_PROVIDER): siliconflow|openai|dashscope|xinference")

    vdb = serve.add_mutually_exclusive_group()
    vdb.add_argument("--vector-db-path", help="Embedded vector DB directory (env QDRANT_PATH)")
    vdb.add_argument("--vector-db-url", help="Vector DB server URL, e.g. http://localhost:6333 (env QDRANT_HOST/PORT)")

    serve.add_argument("--history-db-path", help="Message history sqlite path "
                       "(env HISTORY_DB_PATH, default <vector-db-path>/history.db)")

    serve.add_argument("--enable-bm25", action=argparse.BooleanOptionalAction, default=None,
                       help="BM25 sparse search signal (env ENABLE_BM25, default true)")
    serve.add_argument("--enable-entity", action=argparse.BooleanOptionalAction, default=None,
                       help="Entity extraction and boosting (env ENABLE_ENTITY, default false)")
    serve.add_argument("--rerank", choices=["llm", "cross_encoder", "off"],
                       help="Rerank engine (env RERANK_MODE, default llm); cross_encoder "
                            "uses the CROSS_ENCODER_* env group")
    serve.add_argument("--enable-graph", action=argparse.BooleanOptionalAction, default=None,
                       help="Graph memory store, opt-in (env ENABLE_GRAPH, default false)")

    serve.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=None,
                       help="Enable dedup on write (env DEDUP_ENABLED, default true)")
    serve.add_argument("--dedup-resolver", choices=["skip", "replace", "rewrite", "edit"],
                       help="How to resolve a detected duplicate (env DEDUP_RESOLVER, default rewrite)")
    serve.add_argument("--dedup-detector", choices=["listwise", "listwise_multitarget", "pointwise"],
                       help="How to detect duplicates (env DEDUP_DETECTOR, default listwise_multitarget)")
    serve.add_argument("--dedup-thinking", action=argparse.BooleanOptionalAction, default=None,
                       help="LLM thinking for dedup (env DEDUP_THINKING, default false)")
    serve.add_argument("--dedup-recall-threshold", type=float,
                       help="Candidate recall cutoff for dedup detectors (env DEDUP_RECALL_THRESHOLD, default 0.40)")

    serve.add_argument("--extract-last-k-messages", type=int,
                       help="Extraction context window (env EXTRACT_LAST_K_MESSAGES, default 10)")

    serve.add_argument("--embedder-dims", type=int,
                       help="Embedding dimensions (env EMBEDDER_DIMS, default: auto-detect from API probe)")

    serve.add_argument("--extraction-prompt", help="Custom extraction prompt: prompt file path (env EXTRACTION_PROMPT)")
    serve.add_argument("--dedup-prompt", help="Custom dedup prompt: prompt file path; unset = auto-paired from detector+resolver (env DEDUP_PROMPT)")
    serve.add_argument("--rewrite-prompt", help="Custom merge/rewrite prompt: prompt file path (env REWRITE_PROMPT)")
    serve.add_argument("--edit-prompt", help="Custom patch/edit prompt: prompt file path (env EDIT_PROMPT)")
    serve.add_argument("--rerank-prompt", help="Custom rerank prompt: prompt file path (env LLM_RERANK_PROMPT)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neatmem",
        description="NeatMem — a local mem0-compatible memory server for AI agents",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start the NeatMem server")
    add_serve_arguments(serve)
    sub.add_parser("evaluate", help="Run LOCOMO evaluation (see `neatmem evaluate --help`)")
    sub.add_parser("demo", help="Run a demo case (see `neatmem demo --help`)")
    return parser


def _warn_unknown_neatmem_env() -> None:
    for key in sorted(os.environ):
        if key.startswith("NEATMEM_") and key not in _KNOWN_NEATMEM_ENV:
            print(f"neatmem: warning: unrecognized env var {key} (typo?)", file=sys.stderr)


def serve_flags_to_env(args: argparse.Namespace) -> dict:
    """Translate parsed serve flags to env vars (single source of truth).

    Used by `serve` (inject into os.environ) and `neatmem evaluate` (inject
    into all pipeline child processes — dedup/extraction act at ingest time,
    so flags must become env, not just serve args)."""
    env = {}
    str_flags = {
        "NEATMEM_HOST": args.host,
        "LLM_MODEL": args.llm_model,
        "LLM_API_KEY": args.llm_api_key,
        "OPENAI_BASE_URL": args.llm_base_url,
        "LLM_PROVIDER": args.llm_provider,
        "EMBEDDER_MODEL": args.embedder_model,
        "EMBEDDER_BASE_URL": args.embedder_base_url,
        "EMBEDDER_API_KEY": args.embedder_api_key,
        "EMBEDDER_PROVIDER": args.embedder_provider,
        "HISTORY_DB_PATH": args.history_db_path,
        "DEDUP_RESOLVER": args.dedup_resolver,
        "DEDUP_DETECTOR": args.dedup_detector,
        "EXTRACTION_PROMPT": args.extraction_prompt,
        "DEDUP_PROMPT": args.dedup_prompt,
        "REWRITE_PROMPT": args.rewrite_prompt,
        "EDIT_PROMPT": args.edit_prompt,
        "LLM_RERANK_PROMPT": args.rerank_prompt,
        "RERANK_MODE": args.rerank,
    }
    for env_key, value in str_flags.items():
        if value is not None:
            env[env_key] = value
    # Prompt flags may carry file paths; evaluate spawns children under the
    # output dir, so anchor relative paths at the invocation cwd.
    from neatmem.prompts.loader import absolutize_prompt_value
    for env_key in ("EXTRACTION_PROMPT", "DEDUP_PROMPT", "REWRITE_PROMPT",
                    "EDIT_PROMPT", "LLM_RERANK_PROMPT"):
        if env_key in env:
            env[env_key] = absolutize_prompt_value(env_key, env[env_key])
    if args.port is not None:
        env["NEATMEM_PORT"] = str(args.port)
    if args.extract_last_k_messages is not None:
        env["EXTRACT_LAST_K_MESSAGES"] = str(args.extract_last_k_messages)
    if getattr(args, "dedup_recall_threshold", None) is not None:
        env["DEDUP_RECALL_THRESHOLD"] = str(args.dedup_recall_threshold)
    if args.embedder_dims is not None:
        env["EMBEDDER_DIMS"] = str(args.embedder_dims)

    bool_flags = {
        "ENABLE_BM25": args.enable_bm25,
        "ENABLE_ENTITY": args.enable_entity,
        "ENABLE_GRAPH": args.enable_graph,
        "DEDUP_ENABLED": args.dedup,
        "DEDUP_THINKING": args.dedup_thinking,
    }
    for env_key, value in bool_flags.items():
        if value is not None:
            env[env_key] = "true" if value else "false"

    if args.vector_db_path is not None:
        env["QDRANT_PATH"] = args.vector_db_path
        # Explicit path means embedded mode; clear any inherited host so it
        # does not silently win in config.py (host non-empty -> server mode).
        env["QDRANT_HOST"] = ""
    if args.vector_db_url is not None:
        url = args.vector_db_url if "://" in args.vector_db_url else f"http://{args.vector_db_url}"
        parsed = urlparse(url)
        if not parsed.hostname:
            raise SystemExit(f"neatmem: error: invalid --vector-db-url {args.vector_db_url!r}")
        env["QDRANT_HOST"] = parsed.hostname
        env["QDRANT_PORT"] = str(parsed.port or 6333)
    return env


def _inject(args: argparse.Namespace) -> None:
    os.environ.update(serve_flags_to_env(args))


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
    argv = list(sys.argv[1:] if argv is None else argv)
    # `evaluate` owns its full arg surface (eval flags + passthrough serve
    # flags), so route before the root parser rejects unknown flags.
    if argv and argv[0] == "evaluate":
        from neatmem.evaluation.orchestrator import run_evaluate
        run_evaluate(argv[1:])
        return
    if argv and argv[0] == "demo":
        from neatmem.demo.runner import run_demo
        run_demo(argv[1:])
        return
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        _serve(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
