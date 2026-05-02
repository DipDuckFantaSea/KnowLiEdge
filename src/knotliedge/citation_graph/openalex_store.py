from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from knotliedge.logging_utils.setup import setup_logging

logger = setup_logging()


def now_iso8601() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_openalex_work_id(raw: str) -> Optional[str]:
    """Normalize arbitrary OpenAlex work id strings to canonical HTTPS URL form.

    Args:
        raw: Value from OpenAlex API (often ``https://openalex.org/W…``) or bare ``W…``.

    Returns:
        Canonical ``https://openalex.org/W…`` or None if not parseable.
    """
    s = (raw or "").strip()
    if not s:
        return None
    if s.startswith("https://openalex.org/"):
        tail = s[len("https://openalex.org/") :]
    elif s.startswith("http://openalex.org/"):
        tail = s[len("http://openalex.org/") :]
    else:
        tail = s
    tail = tail.strip().lstrip("/")
    if tail.startswith("W") and len(tail) > 5:
        return f"https://openalex.org/{tail}"
    return None


@dataclass(frozen=True)
class OpenAlexWorkRecord:
    """Row payload for ``openalex_works`` upsert."""

    work_id: str
    doi: Optional[str]
    title: Optional[str]
    publication_year: Optional[int]
    host_venue_display_name: Optional[str]
    authors_json: Optional[str]
    abstract: Optional[str]
    cited_by_count: Optional[int]
    seed_kind: Optional[str]
    depth_seen: Optional[int]
    updated_at: str


@dataclass(frozen=True)
class OpenAlexCiteEdge:
    """Directed citation edge: ``src`` cites ``dst`` (``dst`` is in ``src``'s references)."""

    src_work_id: str
    dst_work_id: str
    source: str
    created_at: str


class OpenAlexCitationStore:
    """SQLite persistence for OpenAlex-derived work nodes and citation edges.

    Lives in the same database file as :class:`CitationGraphStore` ``edges`` table
    (default ``data/05_citation_graph/citations.sqlite3``).
    """

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
                CREATE TABLE IF NOT EXISTS openalex_works (
                    work_id TEXT PRIMARY KEY,
                    doi TEXT,
                    title TEXT,
                    publication_year INTEGER,
                    host_venue_display_name TEXT,
                    authors_json TEXT,
                    abstract TEXT,
                    cited_by_count INTEGER,
                    seed_kind TEXT,
                    depth_seen INTEGER,
                    updated_at TEXT
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS openalex_cite_edges (
                    src_work_id TEXT NOT NULL,
                    dst_work_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT,
                    PRIMARY KEY (src_work_id, dst_work_id, source)
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS openalex_expansion_checkpoint (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_oa_edges_dst ON openalex_cite_edges(dst_work_id);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_oa_edges_src ON openalex_cite_edges(src_work_id);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_oa_works_doi ON openalex_works(doi);")

    def upsert_work(self, rec: OpenAlexWorkRecord) -> None:
        """Insert or replace a single work row."""
        wid = normalize_openalex_work_id(rec.work_id)
        if not wid:
            logger.warning("upsert_work skipped: invalid work_id=%r", rec.work_id)
            return
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO openalex_works(
                    work_id, doi, title, publication_year, host_venue_display_name,
                    authors_json, abstract, cited_by_count, seed_kind, depth_seen, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(work_id) DO UPDATE SET
                    doi=COALESCE(excluded.doi, openalex_works.doi),
                    title=COALESCE(excluded.title, openalex_works.title),
                    publication_year=COALESCE(excluded.publication_year, openalex_works.publication_year),
                    host_venue_display_name=COALESCE(
                        excluded.host_venue_display_name, openalex_works.host_venue_display_name
                    ),
                    authors_json=COALESCE(excluded.authors_json, openalex_works.authors_json),
                    abstract=COALESCE(excluded.abstract, openalex_works.abstract),
                    cited_by_count=COALESCE(excluded.cited_by_count, openalex_works.cited_by_count),
                    seed_kind=COALESCE(excluded.seed_kind, openalex_works.seed_kind),
                    depth_seen=CASE
                        WHEN excluded.depth_seen IS NOT NULL AND openalex_works.depth_seen IS NOT NULL
                            THEN MIN(excluded.depth_seen, openalex_works.depth_seen)
                        WHEN excluded.depth_seen IS NOT NULL THEN excluded.depth_seen
                        ELSE openalex_works.depth_seen
                    END,
                    updated_at=excluded.updated_at;
                """,
                (
                    wid,
                    rec.doi,
                    rec.title,
                    rec.publication_year,
                    rec.host_venue_display_name,
                    rec.authors_json,
                    rec.abstract,
                    rec.cited_by_count,
                    rec.seed_kind,
                    rec.depth_seen,
                    rec.updated_at,
                ),
            )

    def upsert_edges(self, edges: Sequence[OpenAlexCiteEdge]) -> None:
        """Bulk upsert citation edges."""
        if not edges:
            return
        rows: List[Tuple[str, str, str, str]] = []
        for e in edges:
            s = normalize_openalex_work_id(e.src_work_id)
            d = normalize_openalex_work_id(e.dst_work_id)
            if not s or not d:
                continue
            rows.append((s, d, str(e.source), e.created_at))
        if not rows:
            return
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO openalex_cite_edges(src_work_id, dst_work_id, source, created_at)
                VALUES(?,?,?,?)
                ON CONFLICT(src_work_id, dst_work_id, source) DO NOTHING;
                """,
                rows,
            )

    def get_work(self, work_id: str) -> Optional[Dict[str, Any]]:
        """Return one work row as dict, or None."""
        wid = normalize_openalex_work_id(work_id)
        if not wid:
            return None
        with self._connect() as con:
            row = con.execute("SELECT * FROM openalex_works WHERE work_id = ?;", (wid,)).fetchone()
            if row is None:
                return None
            return dict(row)

    def iter_works(self) -> List[Dict[str, Any]]:
        """Load all work rows (intended for small/medium graphs)."""
        out: List[Dict[str, Any]] = []
        with self._connect() as con:
            for row in con.execute("SELECT * FROM openalex_works;"):
                out.append(dict(row))
        return out

    def iter_cite_edges(self) -> List[Tuple[str, str, str]]:
        """Return list of (src_work_id, dst_work_id, source)."""
        out: List[Tuple[str, str, str]] = []
        with self._connect() as con:
            for row in con.execute("SELECT src_work_id, dst_work_id, source FROM openalex_cite_edges;"):
                out.append((str(row["src_work_id"]), str(row["dst_work_id"]), str(row["source"])))
        return out

    def cite_successors(self, work_id: str, *, limit: int = 500) -> List[str]:
        """Works cited by ``work_id`` (outgoing): ``work_id -> dst``."""
        wid = normalize_openalex_work_id(work_id)
        if not wid:
            return []
        out: List[str] = []
        with self._connect() as con:
            for row in con.execute(
                "SELECT DISTINCT dst_work_id FROM openalex_cite_edges WHERE src_work_id = ? LIMIT ?;",
                (wid, int(limit)),
            ):
                out.append(str(row["dst_work_id"]))
        return out

    def cite_predecessors(self, work_id: str, *, limit: int = 500) -> List[str]:
        """Works that cite ``work_id`` (incoming): ``src -> work_id``."""
        wid = normalize_openalex_work_id(work_id)
        if not wid:
            return []
        out: List[str] = []
        with self._connect() as con:
            for row in con.execute(
                "SELECT DISTINCT src_work_id FROM openalex_cite_edges WHERE dst_work_id = ? LIMIT ?;",
                (wid, int(limit)),
            ):
                out.append(str(row["src_work_id"]))
        return out

    def get_checkpoint(self, key: str) -> Optional[Dict[str, Any]]:
        """Load JSON checkpoint blob."""
        k = (key or "").strip()
        if not k:
            return None
        with self._connect() as con:
            row = con.execute(
                "SELECT value_json FROM openalex_expansion_checkpoint WHERE key = ?;", (k,)
            ).fetchone()
            if row is None:
                return None
            raw = row["value_json"]
            if not isinstance(raw, str) or not raw.strip():
                return None
            try:
                obj = json.loads(raw)
            except Exception as e:
                logger.warning("checkpoint JSON parse failed | key=%s | %s", k, e)
                return None
            return obj if isinstance(obj, dict) else None

    def set_checkpoint(self, key: str, payload: Dict[str, Any]) -> None:
        """Persist JSON checkpoint."""
        k = (key or "").strip()
        if not k:
            return
        body = json.dumps(payload, ensure_ascii=False)
        ts = now_iso8601()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO openalex_expansion_checkpoint(key, value_json, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at;
                """,
                (k, body, ts),
            )
