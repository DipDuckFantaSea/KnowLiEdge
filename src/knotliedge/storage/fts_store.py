from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from knotliedge.config.types import AppConfig
from knotliedge.logging_utils.setup import setup_logging

logger = setup_logging()


@dataclass(frozen=True)
class FtsHit:
    chunk_id: str
    score: float
    doc_id: str
    short_name: str
    section: Optional[str]
    preview: str


def default_fts_db_path(cfg: AppConfig) -> Path:
    """Return default FTS5 SQLite path for this project.

    When the vault is under ``sandbox/data/…``, the FTS index is expected at
    ``sandbox/data/04_fts_db/fts.sqlite3`` (same layout as the main tree under
    ``data/04_fts_db/``).

    Args:
        cfg: AppConfig.

    Returns:
        Path to the FTS database file under the project root.
    """
    vault = cfg.paths.markdown_vault_dir
    if "sandbox" in vault.parts:
        return (cfg.project_root / "sandbox" / "data" / "04_fts_db" / "fts.sqlite3").resolve()
    return (cfg.project_root / "data" / "04_fts_db" / "fts.sqlite3").resolve()


class FtsStore:
    """SQLite FTS5 store for keyword search (chunk-level)."""

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
                CREATE TABLE IF NOT EXISTS chunks_meta (
                    chunk_id TEXT PRIMARY KEY,
                    source_md TEXT,
                    chunk_index INTEGER,
                    created_at TEXT
                );
                """
            )
            # Document-level metadata table (async enrichment writes here).
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
                    document_profile TEXT,
                    document_profile_source TEXT,
                    updated_at TEXT
                );
                """
            )
            try:
                cols = {str(r["name"]) for r in con.execute("PRAGMA table_info(documents);").fetchall()}
            except Exception:
                cols = set()
            if "document_profile" not in cols:
                try:
                    con.execute("ALTER TABLE documents ADD COLUMN document_profile TEXT;")
                except Exception:
                    pass
            if "document_profile_source" not in cols:
                try:
                    con.execute("ALTER TABLE documents ADD COLUMN document_profile_source TEXT;")
                except Exception:
                    pass
            # FTS5 virtual table. Using external content is possible, but for MVP we keep it simple.
            con.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    short_name UNINDEXED,
                    section UNINDEXED,
                    text
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_meta_doc_id ON chunks_meta(chunk_id);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_documents_openalex_id ON documents(openalex_id);")

    def upsert_document_profile(
        self,
        *,
        doc_id: str,
        document_profile: str,
        updated_at: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        did = str(doc_id or "").strip()
        if not did:
            return
        prof = str(document_profile or "")
        ts = str(updated_at or "")
        src = str(source or "").strip()
        with self._connect() as con:
            con.execute("INSERT OR IGNORE INTO documents(doc_id) VALUES (?);", (did,))
            con.execute(
                """
                UPDATE documents
                SET document_profile = ?,
                    document_profile_source = CASE WHEN ? != '' THEN ? ELSE document_profile_source END,
                    updated_at = CASE WHEN ? != '' THEN ? ELSE updated_at END
                WHERE doc_id = ?;
                """,
                (prof, src, src, ts, ts, did),
            )

    def get_document_profile(self, *, doc_id: str) -> str:
        did = str(doc_id or "").strip()
        if not did:
            return ""
        with self._connect() as con:
            try:
                row = con.execute("SELECT document_profile FROM documents WHERE doc_id = ? LIMIT 1;", (did,)).fetchone()
            except Exception:
                return ""
        if not row:
            return ""
        return str(row["document_profile"] or "")

    def get_document_profile_source(self, *, doc_id: str) -> str:
        did = str(doc_id or "").strip()
        if not did:
            return ""
        with self._connect() as con:
            try:
                row = con.execute(
                    "SELECT document_profile_source FROM documents WHERE doc_id = ? LIMIT 1;",
                    (did,),
                ).fetchone()
            except Exception:
                return ""
        if not row:
            return ""
        return str(row["document_profile_source"] or "")

    def upsert_chunks(
        self,
        *,
        rows: Sequence[Tuple[str, str, str, Optional[str], str]],
        metas: Optional[Sequence[Tuple[str, str, int, str]]] = None,
        purge_doc_ids: Optional[Iterable[str]] = None,
    ) -> None:
        """Upsert chunk rows into FTS tables.

        Args:
            rows: Sequence of (chunk_id, doc_id, short_name, section, text).
            metas: Optional sequence of (chunk_id, source_md, chunk_index, created_at).
            purge_doc_ids: Optional doc_ids to delete first (best-effort).

        Returns:
            None.
        """
        if not rows and not purge_doc_ids:
            return
        purge = sorted({str(d) for d in (purge_doc_ids or []) if str(d)})
        with self._connect() as con:
            if purge:
                # Delete by doc_id from FTS table. This is fast enough for mid-size local corpora.
                for did in purge:
                    con.execute("DELETE FROM chunks_fts WHERE doc_id = ?;", (did,))
            if rows:
                # Replace semantics for FTS5: delete existing then insert.
                chunk_ids = [r[0] for r in rows if r and r[0]]
                if chunk_ids:
                    con.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?;", [(cid,) for cid in chunk_ids])
                con.executemany(
                    "INSERT INTO chunks_fts(chunk_id, doc_id, short_name, section, text) VALUES(?,?,?,?,?);",
                    [(cid, did, sn, sec, txt) for (cid, did, sn, sec, txt) in rows],
                )
            if metas:
                con.executemany(
                    """
                    INSERT INTO chunks_meta(chunk_id, source_md, chunk_index, created_at)
                    VALUES(?,?,?,?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        source_md=excluded.source_md,
                        chunk_index=excluded.chunk_index,
                        created_at=excluded.created_at;
                    """,
                    [(cid, src, int(idx), str(ts)) for (cid, src, idx, ts) in metas],
                )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        where: Optional[Dict[str, object]] = None,
    ) -> List[FtsHit]:
        """Keyword search via FTS5 BM25.

        Args:
            query: User query string (FTS5 syntax supported).
            top_k: Max hits.
            where: Optional filters: doc_id / short_name.

        Returns:
            List of FtsHit (higher score = better).
        """
        q = (query or "").strip()
        if not q:
            return []

        # FTS5 treats ASCII hyphen between alphanumerics as the NOT operator (e.g. ``GaN-on-Si``).
        # Natural-language queries often contain hyphenated technical tokens; normalize to spaces
        # so BM25 search behaves as users expect. Quoted phrases are left untouched.
        if '"' not in q:
            q = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])", " ", q)

        where = where or {}
        doc_id = str(where.get("doc_id") or "").strip() if where.get("doc_id") is not None else ""
        short_name = str(where.get("short_name") or "").strip() if where.get("short_name") is not None else ""

        clauses: List[str] = ["chunks_fts MATCH ?"]
        params: List[object] = [q]
        if doc_id:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if short_name:
            clauses.append("short_name = ?")
            params.append(short_name)
        sql_where = " AND ".join(clauses)

        sql = f"""
            SELECT
                chunk_id,
                doc_id,
                short_name,
                section,
                substr(text, 1, 400) AS preview,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            WHERE {sql_where}
            ORDER BY bm25_score ASC
            LIMIT ?;
        """
        params.append(int(top_k))

        hits: List[FtsHit] = []
        with self._connect() as con:
            for row in con.execute(sql, params):
                # bm25: smaller is better. Convert to monotonic higher-is-better score.
                try:
                    bm25_score = float(row["bm25_score"])
                except Exception:
                    bm25_score = 0.0
                score = 1.0 / (1.0 + max(0.0, bm25_score))
                hits.append(
                    FtsHit(
                        chunk_id=str(row["chunk_id"] or ""),
                        score=float(score),
                        doc_id=str(row["doc_id"] or ""),
                        short_name=str(row["short_name"] or ""),
                        section=str(row["section"]) if row["section"] is not None else None,
                        preview=str(row["preview"] or ""),
                    )
                )
        return hits

    def search_literal_substring(
        self,
        needle: str,
        *,
        top_k: int = 10,
        where: Optional[Dict[str, object]] = None,
        case_sensitive: bool = True,
        mark_tag: Tuple[str, str] = ("<mark>", "</mark>"),
    ) -> List[FtsHit]:
        """Strict literal substring search against chunk text (with highlight preview).

        This is intentionally different from FTS BM25: it only returns rows where the
        raw chunk text contains the exact `needle` substring. Use this when the user
        explicitly cares about exact token presence (e.g. "GaN on SiN").

        Args:
            needle: Literal substring to match.
            top_k: Max hits.
            where: Optional filters: doc_id / short_name.
            case_sensitive: If True, use case-sensitive match; otherwise case-insensitive.
            mark_tag: Tuple of (open, close) tags for highlighting in preview.

        Returns:
            List of FtsHit (higher score = better). Score is a simple heuristic here.
        """

        n = (needle or "").strip()
        if not n:
            return []

        where = where or {}
        doc_id = str(where.get("doc_id") or "").strip() if where.get("doc_id") is not None else ""
        short_name = str(where.get("short_name") or "").strip() if where.get("short_name") is not None else ""

        # Note: chunks_fts is a virtual table; `text` is stored in the FTS table itself.
        # For strict literal match, we use LIKE/instr on `text`.
        if case_sensitive:
            like_expr = "text LIKE '%' || ? || '%'"
            instr_expr = "instr(text, ?)"
        else:
            like_expr = "lower(text) LIKE '%' || lower(?) || '%'"
            instr_expr = "instr(lower(text), lower(?))"

        clauses: List[str] = [like_expr]
        # Need two bindings for (LIKE ? ...) and instr(..., ?)
        params: List[object] = [n, n]
        if doc_id:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if short_name:
            clauses.append("short_name = ?")
            params.append(short_name)
        sql_where = " AND ".join(clauses)

        sql = f"""
            SELECT
                chunk_id,
                doc_id,
                short_name,
                section,
                text,
                {instr_expr} AS pos
            FROM chunks_fts
            WHERE {sql_where}
            ORDER BY pos ASC
            LIMIT ?;
        """
        params.append(int(top_k))

        pre_open, pre_close = mark_tag
        hits: List[FtsHit] = []
        with self._connect() as con:
            for row in con.execute(sql, params):
                text = str(row["text"] or "")
                pos = int(row["pos"] or 0)
                # Build a preview window around first occurrence.
                start = max(0, pos - 120)
                end = min(len(text), pos + len(n) + 280)
                preview_raw = text[start:end]
                # Highlight the first occurrence in preview only (keep stable output size).
                preview = preview_raw.replace(n, f"{pre_open}{n}{pre_close}", 1) if case_sensitive else preview_raw
                if not case_sensitive:
                    # Best-effort case-insensitive highlight: locate first match by folded search.
                    low = preview_raw.lower()
                    idx = low.find(n.lower())
                    if idx != -1:
                        preview = (
                            preview_raw[:idx]
                            + f"{pre_open}{preview_raw[idx:idx+len(n)]}{pre_close}"
                            + preview_raw[idx + len(n) :]
                        )

                # For strict match, score favors earlier occurrences.
                score = 1.0 / (1.0 + max(0, pos))
                hits.append(
                    FtsHit(
                        chunk_id=str(row["chunk_id"] or ""),
                        score=float(score),
                        doc_id=str(row["doc_id"] or ""),
                        short_name=str(row["short_name"] or ""),
                        section=str(row["section"]) if row["section"] is not None else None,
                        preview=preview.strip(),
                    )
                )
        return hits

    def get_chunk_meta_map(self, chunk_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
        """Fetch stored meta info for chunk ids.

        Args:
            chunk_ids: Chunk ids to query.

        Returns:
            Mapping chunk_id -> metadata dict (may be empty).
        """
        ids = [str(cid) for cid in chunk_ids if str(cid)]
        if not ids:
            return {}
        out: Dict[str, Dict[str, object]] = {}
        with self._connect() as con:
            # SQLite has a variable limit; keep it simple for small lists.
            qmarks = ",".join("?" for _ in ids)
            sql = f"SELECT chunk_id, source_md, chunk_index, created_at FROM chunks_meta WHERE chunk_id IN ({qmarks});"
            for row in con.execute(sql, ids):
                cid = str(row["chunk_id"] or "")
                if not cid:
                    continue
                out[cid] = {
                    "chunk_id": cid,
                    "source_md": str(row["source_md"] or ""),
                    "chunk_index": int(row["chunk_index"] or 0),
                    "created_at": str(row["created_at"] or ""),
                }
        return out

