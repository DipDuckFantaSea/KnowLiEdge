from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from knotliedge.config.types import AppConfig
from knotliedge.logging_utils.setup import setup_logging

logger = setup_logging()


def now_iso8601() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_citation_db_path(cfg: AppConfig) -> Path:
    """Default SQLite path for citation graph DB."""
    if getattr(cfg, "environment", None) is not None and cfg.environment.is_sandbox:
        return (cfg.project_root / "sandbox" / "data" / "05_citation_graph" / "citations.sqlite3").resolve()
    return (cfg.project_root / "data" / "05_citation_graph" / "citations.sqlite3").resolve()


@dataclass(frozen=True)
class Edge:
    src_doc_id: str
    ref_id: str
    ref_type: str
    ref_text: str
    confidence: float
    created_at: str


class CitationGraphStore:
    """SQLite store for citation graph edges."""

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = Path(db_path).resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    src_doc_id TEXT NOT NULL,
                    ref_id TEXT NOT NULL,
                    ref_type TEXT NOT NULL,
                    ref_text TEXT,
                    confidence REAL,
                    created_at TEXT,
                    PRIMARY KEY (src_doc_id, ref_id)
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_edges_ref_id ON edges(ref_id);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_edges_src_doc_id ON edges(src_doc_id);")

    def upsert_edges(self, edges: Sequence[Edge]) -> None:
        if not edges:
            return
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO edges(src_doc_id, ref_id, ref_type, ref_text, confidence, created_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(src_doc_id, ref_id) DO UPDATE SET
                    ref_type=excluded.ref_type,
                    ref_text=excluded.ref_text,
                    confidence=excluded.confidence,
                    created_at=excluded.created_at;
                """,
                [
                    (
                        e.src_doc_id,
                        e.ref_id,
                        e.ref_type,
                        e.ref_text,
                        float(e.confidence),
                        e.created_at,
                    )
                    for e in edges
                ],
            )

    def delete_doc(self, doc_id: str) -> int:
        did = str(doc_id)
        with self._connect() as con:
            cur = con.execute("DELETE FROM edges WHERE src_doc_id = ?;", (did,))
            return int(cur.rowcount or 0)

    def get_references(self, doc_id: str, *, limit: int = 200) -> List[Dict[str, object]]:
        did = str(doc_id)
        out: List[Dict[str, object]] = []
        with self._connect() as con:
            for row in con.execute(
                """
                SELECT ref_id, ref_type, ref_text, confidence, created_at
                FROM edges
                WHERE src_doc_id = ?
                ORDER BY confidence DESC
                LIMIT ?;
                """,
                (did, int(limit)),
            ):
                out.append(dict(row))
        return out

    def get_citations(self, ref_id: str, *, limit: int = 200) -> List[Dict[str, object]]:
        rid = str(ref_id)
        out: List[Dict[str, object]] = []
        with self._connect() as con:
            for row in con.execute(
                """
                SELECT src_doc_id, ref_type, confidence, created_at, substr(ref_text, 1, 400) AS ref_preview
                FROM edges
                WHERE ref_id = ?
                ORDER BY confidence DESC
                LIMIT ?;
                """,
                (rid, int(limit)),
            ):
                out.append(dict(row))
        return out

    def list_doi_reference_seeds(self, *, limit: int = 5000) -> List[str]:
        """Return distinct DOI-like ``ref_id`` values from ``edges`` (``ref_type='doi'``).

        Args:
            limit: Max number of distinct ids to return.

        Returns:
            De-duplicated reference ids (normalized DOI strings as stored in ``edges``).
        """
        out: List[str] = []
        with self._connect() as con:
            for row in con.execute(
                """
                SELECT ref_id
                FROM edges
                WHERE lower(ref_type) = 'doi' AND ref_id IS NOT NULL AND trim(ref_id) != ''
                GROUP BY ref_id
                LIMIT ?;
                """,
                (int(limit),),
            ):
                rid = str(row["ref_id"] or "").strip()
                if rid:
                    out.append(rid)
        return out

    def get_cocitation(self, doc_id: str, *, top_k: int = 10) -> List[Dict[str, object]]:
        """Return docs that share referenced ids with given doc."""
        did = str(doc_id)
        with self._connect() as con:
            # refs that did cites
            ref_rows = con.execute("SELECT ref_id FROM edges WHERE src_doc_id = ?;", (did,)).fetchall()
            ref_ids = [str(r["ref_id"]) for r in ref_rows if r and r["ref_id"]]
            if not ref_ids:
                return []
            qmarks = ",".join("?" for _ in ref_ids)
            sql = f"""
                SELECT src_doc_id, COUNT(*) AS shared_refs
                FROM edges
                WHERE ref_id IN ({qmarks}) AND src_doc_id != ?
                GROUP BY src_doc_id
                ORDER BY shared_refs DESC
                LIMIT ?;
            """
            params: List[object] = list(ref_ids) + [did, int(top_k)]
            out: List[Dict[str, object]] = []
            for row in con.execute(sql, params):
                out.append(dict(row))
            return out

