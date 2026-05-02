from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from knotliedge.chunking.md_chunker import load_markdown_doc
from knotliedge.citation_graph.extract import extract_references_from_markdown
from knotliedge.citation_graph.store import CitationGraphStore, Edge, default_citation_db_path, now_iso8601
from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging

logger = setup_logging()


def run_build_citation_graph(
    *,
    config_path: Path,
    db_path: Optional[Path] = None,
    limit: Optional[int] = None,
    purge_doc: bool = False,
) -> None:
    cfg = load_app_config(Path(config_path))
    store = CitationGraphStore(db_path=Path(db_path) if db_path is not None else default_citation_db_path(cfg))

    md_files = sorted(cfg.paths.markdown_vault_dir.rglob("*.md"))
    if limit is not None:
        md_files = md_files[: int(limit)]
    if not md_files:
        logger.info("No markdown files found under: %s", cfg.paths.markdown_vault_dir)
        return

    created_at = now_iso8601()
    docs_ok = 0
    docs_failed = 0
    edges_total = 0
    for p in md_files:
        try:
            doc = load_markdown_doc(p)
            text = p.read_text(encoding="utf-8", errors="ignore")
            refs = extract_references_from_markdown(text)
            if purge_doc:
                store.delete_doc(str(doc.doc_id))
            edges: List[Edge] = [
                Edge(
                    src_doc_id=str(doc.doc_id),
                    ref_id=r.ref_id,
                    ref_type=r.ref_type,
                    ref_text=r.ref_text,
                    confidence=float(r.confidence),
                    created_at=created_at,
                )
                for r in refs
            ]
            store.upsert_edges(edges)
            docs_ok += 1
            edges_total += len(edges)
        except Exception as e:
            docs_failed += 1
            logger.warning("Citation extract failed: %s | %s", p, e)

    logger.info("Done. docs_ok=%s docs_failed=%s edges_total=%s db=%s", docs_ok, docs_failed, edges_total, store.db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build citation graph SQLite from markdown vault.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional sqlite path for citation graph DB (e.g. sandbox/data/05_citation_graph/citations.sqlite3).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of markdown docs.")
    parser.add_argument("--purge-doc", action="store_true", help="Purge existing edges for each doc before upsert.")
    args = parser.parse_args()
    run_build_citation_graph(
        config_path=Path(args.config),
        db_path=Path(args.db) if args.db else None,
        limit=args.limit,
        purge_doc=bool(args.purge_doc),
    )


if __name__ == "__main__":
    main()

