from __future__ import annotations

import argparse
from pathlib import Path

from knotliedge.config.load import load_app_config
from knotliedge.config.paths import ensure_dirs
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.pipeline.index_markdown import run_index_markdown


def main() -> None:
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Index Markdown vault into ChromaDB.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of markdown docs to process.")
    parser.add_argument(
        "--enable-fts",
        action="store_true",
        help="Also index chunks into a local SQLite FTS5 database (keyword search).",
    )
    parser.add_argument(
        "--fts-db",
        type=str,
        default=None,
        help="Optional path to FTS5 sqlite DB. Default: data/04_fts_db/fts.sqlite3 under project root.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["full", "incremental", "purge-missing"],
        help=(
            "Indexing mode. If set, it overrides some legacy flags: "
            "full=index all docs with per-doc purge; "
            "incremental=skip unchanged docs by doc_hash+mtime with per-doc purge; "
            "purge-missing=delete doc_ids that no longer exist in vault (no indexing)."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the Chroma collection before indexing (full rebuild).",
    )
    parser.add_argument(
        "--purge-doc",
        action="store_true",
        help="Before indexing each doc, delete existing chunks for the same doc_id (incremental safe).",
    )
    parser.add_argument(
        "--rebuild-hard",
        action="store_true",
        help="Hard rebuild: delete ALL files under chroma_db_dir before rebuilding the collection.",
    )
    args = parser.parse_args()

    cfg = load_app_config(Path(args.config))
    ensure_dirs(cfg)

    stats = run_index_markdown(
        cfg,
        config_path=Path(args.config),
        mode=args.mode,
        limit=args.limit,
        rebuild=bool(args.rebuild),
        purge_doc=bool(args.purge_doc),
        rebuild_hard=bool(args.rebuild_hard),
        enable_fts=bool(args.enable_fts),
        fts_db_path=Path(args.fts_db) if args.fts_db else None,
    )
    logger.info(
        "Done. docs_total=%s docs_ok=%s docs_skipped=%s docs_failed=%s chunks_total=%s",
        stats.total_docs,
        stats.succeeded_docs,
        stats.skipped_docs,
        stats.failed_docs,
        stats.total_chunks,
    )


if __name__ == "__main__":
    main()

