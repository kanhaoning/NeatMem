"""CLI for the scoped Qdrant archive export/import commands."""

import argparse
import sys
from typing import List

from neatmem.migration.qdrant_archive import (
    QdrantArchiveError,
    export_collection,
    import_collection,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _cmd_export(args: argparse.Namespace) -> int:
    try:
        result = export_collection(
            source_path=args.source_path,
            collection_name=args.collection,
            output_path=args.output,
            batch_size=args.batch_size,
        )
    except QdrantArchiveError as exc:
        print(f"neatmem migrate qdrant export: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Exported {result['points']} points from collection "
        f"{result['collection']!r} to {result['output']}"
    )
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    try:
        result = import_collection(
            input_path=args.input,
            target_url=args.target_url,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    except QdrantArchiveError as exc:
        print(f"neatmem migrate qdrant import: error: {exc}", file=sys.stderr)
        return 1
    if result["dry_run"]:
        print(
            f"Archive validated: {result['points']} points for collection "
            f"{result['collection']!r}; no data was written"
        )
    else:
        print(
            f"Imported and verified {result['points']} points into collection "
            f"{result['collection']!r}"
        )
    return 0


def run_migrate(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(prog="neatmem migrate")
    sub = parser.add_subparsers(dest="migration_command", required=True)
    qdrant = sub.add_parser("qdrant", help="Export/import Qdrant point archives")
    qdrant_sub = qdrant.add_subparsers(dest="qdrant_command", required=True)

    export = qdrant_sub.add_parser(
        "export", help="Export an embedded Qdrant collection to a ZIP archive"
    )
    export.add_argument("--source-path", required=True, help="Embedded Qdrant storage directory")
    export.add_argument("--collection", required=True, help="Source collection name")
    export.add_argument("--output", required=True, help="New ZIP archive path (must not exist)")
    export.add_argument("--batch-size", type=_positive_int, default=100)
    export.set_defaults(handler=_cmd_export)

    import_parser = qdrant_sub.add_parser(
        "import", help="Import a Qdrant ZIP archive into a new server collection"
    )
    import_parser.add_argument("--input", required=True, help="Qdrant ZIP archive path")
    import_parser.add_argument(
        "--target-url", required=True, help="Qdrant server URL, e.g. http://127.0.0.1:6333"
    )
    import_parser.add_argument("--batch-size", type=_positive_int, default=100)
    import_parser.add_argument(
        "--dry-run", action="store_true", help="Validate without creating or writing a collection"
    )
    import_parser.set_defaults(handler=_cmd_import)

    args = parser.parse_args(argv)
    sys.exit(args.handler(args))
