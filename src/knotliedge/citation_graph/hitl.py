from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from knotliedge.citation_graph.openalex_store import (
    OpenAlexCitationStore,
    OpenAlexCiteEdge,
    OpenAlexWorkRecord,
    normalize_openalex_work_id,
    now_iso8601,
)
from knotliedge.logging_utils.setup import setup_logging

logger = setup_logging()


@dataclass(frozen=True)
class StagedCounts:
    run_id: str
    pending_works: int
    pending_edges: int
    approved_works: int
    approved_edges: int
    rejected_works: int
    rejected_edges: int


class OpenAlexExpansionStagingStore:
    """Staging area for OpenAlex citation expansion (human-in-the-loop).

    This store persists candidates into the *same* SQLite file as the formal
    OpenAlex tables, but in separate staging tables. Only an explicit approve
    action will materialize them into ``openalex_works`` / ``openalex_cite_edges``.
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
                CREATE TABLE IF NOT EXISTS openalex_expansion_stage_items (
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL, -- "work" or "edge"
                    work_id TEXT,
                    src_work_id TEXT,
                    dst_work_id TEXT,
                    source TEXT,
                    payload_json TEXT,
                    created_at TEXT,
                    status TEXT NOT NULL DEFAULT 'pending', -- pending/approved/rejected
                    PRIMARY KEY (run_id, kind, COALESCE(work_id,''), COALESCE(src_work_id,''), COALESCE(dst_work_id,''), COALESCE(source,''))
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS openalex_expansion_stage_checkpoint (
                    run_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at TEXT,
                    PRIMARY KEY (run_id, key)
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_oa_stage_status ON openalex_expansion_stage_items(status);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_oa_stage_kind ON openalex_expansion_stage_items(kind);")

    def stage_work(self, *, run_id: str, rec: OpenAlexWorkRecord) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            raise ValueError("run_id is required")
        wid = normalize_openalex_work_id(rec.work_id)
        if not wid:
            return
        payload = json.dumps(rec.__dict__, ensure_ascii=False)
        ts = now_iso8601()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO openalex_expansion_stage_items(run_id, kind, work_id, payload_json, created_at, status)
                VALUES(?,?,?,?,?, 'pending')
                ON CONFLICT(run_id, kind, COALESCE(work_id,''), COALESCE(src_work_id,''), COALESCE(dst_work_id,''), COALESCE(source,'')) DO NOTHING;
                """,
                (rid, "work", wid, payload, ts),
            )

    def stage_edges(self, *, run_id: str, edges: Sequence[OpenAlexCiteEdge]) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            raise ValueError("run_id is required")
        rows: List[Tuple[str, str, str, str, str]] = []
        ts = now_iso8601()
        for e in edges or []:
            s = normalize_openalex_work_id(e.src_work_id)
            d = normalize_openalex_work_id(e.dst_work_id)
            if not s or not d:
                continue
            rows.append((rid, s, d, str(e.source), ts))
        if not rows:
            return
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO openalex_expansion_stage_items(run_id, kind, src_work_id, dst_work_id, source, created_at, status)
                VALUES(?, 'edge', ?, ?, ?, ?, 'pending')
                ON CONFLICT(run_id, kind, COALESCE(work_id,''), COALESCE(src_work_id,''), COALESCE(dst_work_id,''), COALESCE(source,'')) DO NOTHING;
                """,
                rows,
            )

    def get_checkpoint(self, *, run_id: str, key: str) -> Optional[Dict[str, Any]]:
        rid = str(run_id or "").strip()
        k = str(key or "").strip()
        if not rid or not k:
            return None
        with self._connect() as con:
            row = con.execute(
                "SELECT value_json FROM openalex_expansion_stage_checkpoint WHERE run_id=? AND key=?;",
                (rid, k),
            ).fetchone()
            if row is None:
                return None
            raw = row["value_json"]
            if not isinstance(raw, str) or not raw.strip():
                return None
            try:
                obj = json.loads(raw)
            except Exception:
                return None
            return obj if isinstance(obj, dict) else None

    def set_checkpoint(self, *, run_id: str, key: str, payload: Dict[str, Any]) -> None:
        rid = str(run_id or "").strip()
        k = str(key or "").strip()
        if not rid or not k:
            return
        body = json.dumps(payload, ensure_ascii=False)
        ts = now_iso8601()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO openalex_expansion_stage_checkpoint(run_id, key, value_json, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(run_id, key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at;
                """,
                (rid, k, body, ts),
            )

    def counts(self, *, run_id: str) -> StagedCounts:
        rid = str(run_id or "").strip()
        if not rid:
            raise ValueError("run_id is required")
        with self._connect() as con:
            def _count(kind: str, status: str) -> int:
                row = con.execute(
                    "SELECT COUNT(1) AS c FROM openalex_expansion_stage_items WHERE run_id=? AND kind=? AND status=?;",
                    (rid, kind, status),
                ).fetchone()
                return int(row["c"] or 0) if row is not None else 0

            return StagedCounts(
                run_id=rid,
                pending_works=_count("work", "pending"),
                pending_edges=_count("edge", "pending"),
                approved_works=_count("work", "approved"),
                approved_edges=_count("edge", "approved"),
                rejected_works=_count("work", "rejected"),
                rejected_edges=_count("edge", "rejected"),
            )

    def approve_all_pending(self, *, run_id: str, limit: int = 5000) -> Dict[str, Any]:
        """Materialize pending staged items into formal OpenAlex tables."""

        rid = str(run_id or "").strip()
        if not rid:
            raise ValueError("run_id is required")
        oa = OpenAlexCitationStore(db_path=self._db_path)
        works: List[OpenAlexWorkRecord] = []
        edges: List[OpenAlexCiteEdge] = []
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT kind, work_id, src_work_id, dst_work_id, source, payload_json, created_at
                FROM openalex_expansion_stage_items
                WHERE run_id=? AND status='pending'
                ORDER BY created_at ASC
                LIMIT ?;
                """,
                (rid, int(limit)),
            ).fetchall()
            for row in rows:
                kind = str(row["kind"] or "")
                if kind == "work":
                    raw = row["payload_json"]
                    if isinstance(raw, str) and raw.strip():
                        try:
                            obj = json.loads(raw)
                            if isinstance(obj, dict):
                                works.append(OpenAlexWorkRecord(**obj))  # type: ignore[arg-type]
                        except Exception:
                            continue
                elif kind == "edge":
                    s = str(row["src_work_id"] or "")
                    d = str(row["dst_work_id"] or "")
                    src = str(row["source"] or "")
                    ts = str(row["created_at"] or now_iso8601())
                    if s and d and src:
                        edges.append(OpenAlexCiteEdge(src_work_id=s, dst_work_id=d, source=src, created_at=ts))

        for w in works:
            oa.upsert_work(w)
        if edges:
            oa.upsert_edges(edges)

        ts2 = now_iso8601()
        with self._connect() as con:
            con.execute(
                """
                UPDATE openalex_expansion_stage_items
                SET status='approved'
                WHERE run_id=? AND status='pending' AND created_at <= ?;
                """,
                (rid, ts2),
            )
        return {
            "run_id": rid,
            "approved_works": len(works),
            "approved_edges": len(edges),
        }

