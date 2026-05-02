from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from knotliedge.config.types import AppConfig
from knotliedge.logging_utils.setup import setup_logging

logger = setup_logging()


def default_venue_radar_db_path(cfg: AppConfig) -> Path:
    """Return default SQLite path for venue radar metadata.

    This DB is intentionally isolated from the FTS5 DB used by the core knowledge base.

    Args:
        cfg: AppConfig.

    Returns:
        Path to the radar sqlite database file under the project root.
    """

    if getattr(cfg, "environment", None) is not None and cfg.environment.is_sandbox:
        return (cfg.project_root / "sandbox" / "data" / "07_venue_radar" / "venue_radar.sqlite3").resolve()
    return (cfg.project_root / "data" / "07_venue_radar" / "venue_radar.sqlite3").resolve()


class VenueRadarStore:
    """SQLite store for OpenAlex venue abstracts (quarantine metadata)."""

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
                CREATE TABLE IF NOT EXISTS venue_abstracts (
                    id TEXT PRIMARY KEY,
                    openalex_id TEXT,
                    doi TEXT,
                    title TEXT,
                    abstract TEXT,
                    publication_date TEXT,
                    venue_name TEXT,
                    url TEXT,
                    authors_json TEXT,
                    institutions_json TEXT,
                    in_local_vault INTEGER,
                    local_doc_id TEXT,
                    local_short_name TEXT,
                    local_md_path TEXT,
                    local_match TEXT,
                    local_marked_at TEXT
                );
                """
            )

            # Backward-compatible migration for existing DBs created before new columns.
            try:
                cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(venue_abstracts);")}
            except Exception:
                cols = set()
            if "openalex_id" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN openalex_id TEXT;")
            if "doi" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN doi TEXT;")
            if "authors_json" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN authors_json TEXT;")
            if "institutions_json" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN institutions_json TEXT;")
            if "in_local_vault" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN in_local_vault INTEGER;")
            if "local_doc_id" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN local_doc_id TEXT;")
            if "local_short_name" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN local_short_name TEXT;")
            if "local_md_path" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN local_md_path TEXT;")
            if "local_match" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN local_match TEXT;")
            if "local_marked_at" not in cols:
                con.execute("ALTER TABLE venue_abstracts ADD COLUMN local_marked_at TEXT;")

            # Indices (must be created after any ALTER TABLE migration above).
            con.execute("CREATE INDEX IF NOT EXISTS idx_venue_abstracts_date ON venue_abstracts(publication_date);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_venue_abstracts_venue ON venue_abstracts(venue_name);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_venue_abstracts_openalex_id ON venue_abstracts(openalex_id);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_venue_abstracts_doi ON venue_abstracts(doi);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_venue_abstracts_in_local ON venue_abstracts(in_local_vault);")

    def upsert_abstract(
        self,
        *,
        id: str,
        openalex_id: Optional[str] = None,
        doi: Optional[str] = None,
        title: str,
        abstract: str,
        publication_date: Optional[str],
        venue_name: Optional[str],
        url: Optional[str],
        authors_json: Optional[str] = None,
        institutions_json: Optional[str] = None,
    ) -> None:
        """Upsert one venue abstract row.

        Args:
            id: Stable id (recommend: OpenAlex work id or derived key).
            title: Work title.
            abstract: Work abstract (plain text).
            publication_date: ISO date string (YYYY-MM-DD) if available.
            venue_name: Venue display name if available.
            url: Work URL if available.
            authors_json: JSON-encoded author display names (optional).
            institutions_json: JSON-encoded institution display names (optional).

        Returns:
            None.
        """

        id_s = str(id or "").strip()
        if not id_s:
            raise ValueError("id must be non-empty")
        oa = str(openalex_id or "").strip() or id_s
        doi_s = str(doi or "").strip() if doi else ""
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO venue_abstracts(
                    id, openalex_id, doi, title, abstract, publication_date, venue_name, url, authors_json, institutions_json
                )
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    openalex_id=excluded.openalex_id,
                    doi=excluded.doi,
                    title=excluded.title,
                    abstract=excluded.abstract,
                    publication_date=excluded.publication_date,
                    venue_name=excluded.venue_name,
                    url=excluded.url,
                    authors_json=excluded.authors_json,
                    institutions_json=excluded.institutions_json;
                """,
                (
                    id_s,
                    oa,
                    doi_s or None,
                    str(title or ""),
                    str(abstract or ""),
                    str(publication_date or "") if publication_date else None,
                    str(venue_name or "") if venue_name else None,
                    str(url or "") if url else None,
                    str(authors_json or "") if authors_json else None,
                    str(institutions_json or "") if institutions_json else None,
                ),
            )

    def get_abstracts_by_ids(self, ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
        """Fetch venue_abstracts rows by ids.

        Args:
            ids: Iterable of ids.

        Returns:
            Mapping id -> row dict (may be empty if not found).
        """

        q_ids: List[str] = []
        seen = set()
        for x in ids:
            s = str(x or "").strip()
            if not s:
                continue
            k = s.casefold()
            if k in seen:
                continue
            seen.add(k)
            q_ids.append(s)
        if not q_ids:
            return {}

        out: Dict[str, Dict[str, object]] = {}
        with self._connect() as con:
            qmarks = ",".join("?" for _ in q_ids)
            sql = f"""
                SELECT
                    id, openalex_id, doi, title, abstract, publication_date, venue_name, url,
                    authors_json, institutions_json,
                    in_local_vault, local_doc_id, local_short_name, local_md_path, local_match, local_marked_at
                FROM venue_abstracts
                WHERE id IN ({qmarks});
            """
            for row in con.execute(sql, q_ids):
                rid = str(row["id"] or "")
                if not rid:
                    continue
                in_local_raw = row["in_local_vault"]
                try:
                    in_local = bool(int(in_local_raw or 0))
                except Exception:
                    in_local = False
                out[rid] = {
                    "id": rid,
                    "openalex_id": str(row["openalex_id"] or ""),
                    "doi": str(row["doi"] or ""),
                    "title": str(row["title"] or ""),
                    "abstract": str(row["abstract"] or ""),
                    "publication_date": str(row["publication_date"] or ""),
                    "venue_name": str(row["venue_name"] or ""),
                    "url": str(row["url"] or ""),
                    "authors": _safe_load_json_list(row["authors_json"]),
                    "institutions": _safe_load_json_list(row["institutions_json"]),
                    "in_local_vault": in_local,
                    "local_doc_id": str(row["local_doc_id"] or ""),
                    "local_short_name": str(row["local_short_name"] or ""),
                    "local_md_path": str(row["local_md_path"] or ""),
                    "local_match": str(row["local_match"] or ""),
                    "local_marked_at": str(row["local_marked_at"] or ""),
                }
        return out

    def mark_local_presence(
        self,
        *,
        matches: Dict[str, Dict[str, str]],
        marked_at: str,
    ) -> int:
        """Update `in_local_vault` flags for radar rows.

        Args:
            matches: Mapping radar_id -> dict with optional keys:
                - local_doc_id
                - local_short_name
                - local_md_path
                - local_match
            marked_at: ISO timestamp to record for the scan.

        Returns:
            Number of rows updated (best-effort).
        """

        if not matches:
            return 0
        rows = []
        for rid, info in matches.items():
            r = str(rid or "").strip()
            if not r:
                continue
            d = info if isinstance(info, dict) else {}
            rows.append(
                (
                    1,
                    str(d.get("local_doc_id") or ""),
                    str(d.get("local_short_name") or ""),
                    str(d.get("local_md_path") or ""),
                    str(d.get("local_match") or ""),
                    str(marked_at or ""),
                    r,
                )
            )
        if not rows:
            return 0
        with self._connect() as con:
            cur = con.executemany(
                """
                UPDATE venue_abstracts
                SET
                    in_local_vault=?,
                    local_doc_id=?,
                    local_short_name=?,
                    local_md_path=?,
                    local_match=?,
                    local_marked_at=?
                WHERE id=?;
                """,
                rows,
            )
            return int(cur.rowcount or 0)

    def mark_local_scan(
        self,
        *,
        radar_ids: Sequence[str],
        matches: Dict[str, Dict[str, str]],
        marked_at: str,
    ) -> int:
        """Mark local-vault presence for a scan batch (writes marked_at for all ids).

        Unlike :meth:`mark_local_presence`, this method will write ``local_marked_at`` for
        every ``radar_id`` in the batch, setting ``in_local_vault`` to 1 when a match exists,
        otherwise 0.

        Args:
            radar_ids: Radar ids processed in this scan batch.
            matches: Mapping radar_id -> local match info (same keys as mark_local_presence).
            marked_at: ISO timestamp to record for this scan batch.

        Returns:
            Number of rows updated (best-effort).
        """

        ids: List[str] = []
        seen = set()
        for rid in radar_ids or []:
            s = str(rid or "").strip()
            if not s:
                continue
            k = s.casefold()
            if k in seen:
                continue
            seen.add(k)
            ids.append(s)
        if not ids:
            return 0

        rows = []
        m = matches if isinstance(matches, dict) else {}
        for r in ids:
            info = m.get(r) if isinstance(m.get(r), dict) else {}
            is_local = 1 if (r in m) else 0
            rows.append(
                (
                    int(is_local),
                    str(info.get("local_doc_id") or ""),
                    str(info.get("local_short_name") or ""),
                    str(info.get("local_md_path") or ""),
                    str(info.get("local_match") or ""),
                    str(marked_at or ""),
                    r,
                )
            )
        with self._connect() as con:
            cur = con.executemany(
                """
                UPDATE venue_abstracts
                SET
                    in_local_vault=?,
                    local_doc_id=?,
                    local_short_name=?,
                    local_md_path=?,
                    local_match=?,
                    local_marked_at=?
                WHERE id=?;
                """,
                rows,
            )
            return int(cur.rowcount or 0)

    def purge_all(self) -> int:
        """Delete all rows from venue_abstracts.

        Returns:
            Number of deleted rows.
        """

        with self._connect() as con:
            cur = con.execute("DELETE FROM venue_abstracts;")
            return int(cur.rowcount or 0)

    def get_stats(self, *, top_n: int = 20) -> Dict[str, Any]:
        """Return basic stats and duplicate signals for venue radar rows.

        Args:
            top_n: Max rows to return for duplicate lists.

        Returns:
            Stats dict.
        """

        n = max(0, int(top_n))
        with self._connect() as con:
            row = con.execute("SELECT COUNT(1) AS n FROM venue_abstracts;").fetchone()
            row_count = int(row["n"] if row is not None else 0)
            row = con.execute("SELECT MAX(publication_date) AS v FROM venue_abstracts;").fetchone()
            max_pub_date = (row["v"] if row is not None else None)
            row = con.execute("SELECT MAX(local_marked_at) AS v FROM venue_abstracts;").fetchone()
            max_marked_at = (row["v"] if row is not None else None)

            dup_doi_rows = list(
                con.execute(
                    """
                    SELECT doi, COUNT(1) AS n
                    FROM venue_abstracts
                    WHERE doi IS NOT NULL AND TRIM(doi) <> ''
                    GROUP BY doi
                    HAVING COUNT(1) > 1
                    ORDER BY n DESC
                    LIMIT ?;
                    """,
                    (n,),
                ).fetchall()
            )
            dup_title_rows = list(
                con.execute(
                    """
                    SELECT LOWER(TRIM(title)) AS t, COUNT(1) AS n
                    FROM venue_abstracts
                    WHERE title IS NOT NULL AND TRIM(title) <> ''
                    GROUP BY LOWER(TRIM(title))
                    HAVING COUNT(1) > 1
                    ORDER BY n DESC
                    LIMIT ?;
                    """,
                    (n,),
                ).fetchall()
            )
            return {
                "db_path": str(self._db_path),
                "row_count": row_count,
                "max_publication_date": max_pub_date,
                "max_local_marked_at": max_marked_at,
                "duplicate_doi_top": [{"doi": str(r["doi"] or ""), "count": int(r["n"] or 0)} for r in dup_doi_rows],
                "duplicate_title_top": [
                    {"title_norm": str(r["t"] or ""), "count": int(r["n"] or 0)} for r in dup_title_rows
                ],
            }


def _safe_load_json_list(raw: object) -> List[str]:
    if raw is None:
        return []
    s = str(raw or "").strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: List[str] = []
    for x in obj:
        if x is None:
            continue
        t = str(x).strip()
        if t:
            out.append(t)
    return out

