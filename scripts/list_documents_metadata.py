from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.storage.fts_store import default_fts_db_path

logger = logging.getLogger(__name__)


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def _ensure_documents_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            doi TEXT,
            openalex_id TEXT,
            citation_count INTEGER,
            publication_year INTEGER,
            journal_name TEXT,
            openalex_title TEXT,
            openalex_authors_json TEXT,
            updated_at TEXT
        );
        """
    )
    # Best-effort add columns for older DBs.
    try:
        cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(documents);").fetchall()}
    except Exception:
        cols = set()
    if "openalex_title" not in cols:
        try:
            con.execute("ALTER TABLE documents ADD COLUMN openalex_title TEXT;")
        except Exception:
            pass
    if "openalex_authors_json" not in cols:
        try:
            con.execute("ALTER TABLE documents ADD COLUMN openalex_authors_json TEXT;")
        except Exception:
            pass


def _iter_doc_ids(con: sqlite3.Connection) -> List[str]:
    rows = con.execute(
        "SELECT DISTINCT doc_id FROM chunks_fts WHERE doc_id IS NOT NULL AND doc_id != '' ORDER BY doc_id ASC;"
    ).fetchall()
    return [str(r["doc_id"]) for r in rows if r and r["doc_id"]]


def _seed_documents(con: sqlite3.Connection, doc_ids: Sequence[str]) -> None:
    con.executemany("INSERT OR IGNORE INTO documents(doc_id) VALUES (?);", [(d,) for d in doc_ids if d])


def _get_source_md_for_doc(con: sqlite3.Connection, doc_id: str) -> Optional[Path]:
    row = con.execute(
        """
        SELECT source_md
        FROM chunks_meta
        WHERE chunk_id IN (SELECT chunk_id FROM chunks_fts WHERE doc_id = ? LIMIT 1)
        LIMIT 1;
        """,
        (str(doc_id),),
    ).fetchone()
    if not row:
        return None
    p = str(row["source_md"] or "").strip()
    return Path(p) if p else None


def _load_frontmatter(md_path: Path) -> Dict[str, Any]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}
    fm_raw = "\n".join(lines[1:end])
    fm = yaml.safe_load(fm_raw) or {}
    return fm if isinstance(fm, dict) else {}


@dataclass(frozen=True)
class DocMetaRow:
    doc_id: str
    title: Optional[str]
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    openalex_id: Optional[str]
    citation_count: Optional[int]
    publication_year: Optional[int]
    journal_name: Optional[str]
    source_md: Optional[str]
    source_pdf: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "openalex_id": self.openalex_id,
            "citation_count": self.citation_count,
            "publication_year": self.publication_year,
            "journal_name": self.journal_name,
            "source_md": self.source_md,
            "source_pdf": self.source_pdf,
        }


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="List document-level metadata (sandbox-first). Outputs JSONL.")
    parser.add_argument("--config", type=str, default="sandbox/configs/sandbox.yaml", help="Config path.")
    parser.add_argument("--fts-db", type=str, default=None, help="FTS sqlite path override.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs.")
    parser.add_argument(
        "--utf8",
        action="store_true",
        help="Output UTF-8 JSON (may break on legacy Windows consoles). Default is ASCII-escaped JSON.",
    )
    args = parser.parse_args()

    cfg = load_app_config(Path(args.config))
    fts_db = Path(args.fts_db).resolve() if args.fts_db else default_fts_db_path(cfg)

    if not fts_db.exists():
        raise FileNotFoundError(f"FTS sqlite not found: {fts_db}")

    with _connect(fts_db) as con:
        _ensure_documents_table(con)
        doc_ids = _iter_doc_ids(con)
        _seed_documents(con, doc_ids)

        sql = """
            SELECT doc_id, doi, openalex_id, citation_count, publication_year, journal_name, openalex_title, openalex_authors_json
            FROM documents
            ORDER BY doc_id ASC
        """
        if args.limit is not None:
            sql += " LIMIT ?"
            rows = con.execute(sql, (int(args.limit),)).fetchall()
        else:
            rows = con.execute(sql).fetchall()

        out_rows: List[DocMetaRow] = []
        for r in rows:
            did = str(r["doc_id"])
            md_path = _get_source_md_for_doc(con, did)
            fm: Dict[str, Any] = {}
            if md_path and md_path.exists():
                try:
                    fm = _load_frontmatter(md_path)
                except Exception as e:
                    logger.warning("Failed to parse frontmatter | doc_id=%s md=%s | %s", did, md_path, e)
                    fm = {}

            title = str(fm.get("title") or "").strip() or None
            authors_raw = fm.get("authors") or []
            authors: List[str] = []
            if isinstance(authors_raw, list):
                authors = [str(a).strip() for a in authors_raw if str(a).strip()]
            if not title:
                t2 = r["openalex_title"]
                title = str(t2).strip() if t2 is not None and str(t2).strip() else None
            if not authors:
                try:
                    raw = r["openalex_authors_json"]
                    if raw is not None and str(raw).strip():
                        parsed = json.loads(str(raw))
                        if isinstance(parsed, list):
                            authors = [str(a).strip() for a in parsed if str(a).strip()]
                except Exception:
                    pass
            year = fm.get("year")
            try:
                year_i = int(year) if year is not None and str(year).strip() else None
            except Exception:
                year_i = None

            out_rows.append(
                DocMetaRow(
                    doc_id=did,
                    title=title,
                    authors=authors,
                    year=year_i,
                    doi=str(r["doi"]).strip() if r["doi"] is not None and str(r["doi"]).strip() else None,
                    openalex_id=str(r["openalex_id"]).strip()
                    if r["openalex_id"] is not None and str(r["openalex_id"]).strip()
                    else None,
                    citation_count=int(r["citation_count"]) if r["citation_count"] is not None else None,
                    publication_year=int(r["publication_year"]) if r["publication_year"] is not None else None,
                    journal_name=str(r["journal_name"]).strip()
                    if r["journal_name"] is not None and str(r["journal_name"]).strip()
                    else None,
                    source_md=str(md_path) if md_path is not None else None,
                    source_pdf=str(fm.get("source_pdf") or "").strip() or None,
                )
            )

        for row in out_rows:
            print(json.dumps(row.to_dict(), ensure_ascii=not bool(args.utf8)))


if __name__ == "__main__":
    main()

