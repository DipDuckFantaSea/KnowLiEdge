from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from knotliedge.chunking.md_chunker import chunk_markdown, load_markdown_doc
from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.storage.fts_store import FtsStore


def run_build_fts(*, config_path: Path, fts_db_path: Path, limit: Optional[int] = None) -> None:
    logger = setup_logging()
    cfg = load_app_config(Path(config_path))
    store = FtsStore(db_path=Path(fts_db_path))

    md_files = sorted(cfg.paths.markdown_vault_dir.rglob("*.md"))
    if limit is not None:
        md_files = md_files[: int(limit)]
    if not md_files:
        logger.info("No markdown files found under: %s", cfg.paths.markdown_vault_dir)
        return

    total_chunks = 0
    docs_ok = 0
    docs_failed = 0

    for md_path in tqdm(md_files, desc="build_fts", ascii=True):
        try:
            doc = load_markdown_doc(md_path)
            chunks = chunk_markdown(doc, cfg.chunking)
            if not chunks:
                continue

            # Purge + upsert per doc_id (idempotent rebuild).
            store.upsert_chunks(rows=[], purge_doc_ids=[str(doc.doc_id)])
            rows = [(c.chunk_id, c.doc_id, doc.short_name, c.section, c.text) for c in chunks]
            metas = [(c.chunk_id, str(c.source_md.resolve()), int(c.chunk_index), "") for c in chunks]
            store.upsert_chunks(rows=rows, metas=metas)

            docs_ok += 1
            total_chunks += len(chunks)
        except Exception as e:
            docs_failed += 1
            logger.warning("FTS build failed: %s | %s", md_path, e)

    logger.info("Done. docs_ok=%s docs_failed=%s chunks_total=%s fts_db=%s", docs_ok, docs_failed, total_chunks, store.db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/rebuild SQLite FTS5 index from markdown vault (no embeddings).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--fts-db", type=str, required=True, help="FTS sqlite path (e.g. sandbox/data/04_fts_db/fts.sqlite3).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of markdown docs.")
    args = parser.parse_args()
    run_build_fts(config_path=Path(args.config), fts_db_path=Path(args.fts_db), limit=args.limit)


if __name__ == "__main__":
    main()

