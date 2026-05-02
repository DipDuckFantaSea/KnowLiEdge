from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from knotliedge.chunking.md_chunker import MarkdownDoc, load_markdown_doc, split_frontmatter
from knotliedge.config.load import load_app_config
from knotliedge.metadata.doi_extractor import extract_or_find_doi
from knotliedge.metadata.openalex_client import fetch_openalex_metadata
from knotliedge.storage.fts_store import default_fts_db_path

logger = logging.getLogger(__name__)


def default_manifest_path(cfg_root: Path) -> Path:
    return (Path(cfg_root).resolve() / "sandbox" / "data" / "06_metadata" / "openalex_enrich_manifest.jsonl").resolve()


def load_manifest(path: Path) -> Dict[str, Dict[str, object]]:
    p = Path(path).resolve()
    if not p.exists():
        return {}
    out: Dict[str, Dict[str, object]] = {}
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            did = str(obj.get("doc_id") or "").strip()
            if not did:
                continue
            out[did] = obj
    except Exception as e:
        logger.warning("Failed to load manifest | path=%s | %s", p, e)
        return {}
    return out


def save_manifest(path: Path, manifest: Dict[str, Dict[str, object]]) -> None:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for did in sorted(manifest.keys()):
        obj = manifest.get(did) or {}
        if not isinstance(obj, dict):
            continue
        obj = dict(obj)
        obj["doc_id"] = did
        lines.append(json.dumps(obj, ensure_ascii=False))
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def now_iso8601() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def manifest_force_set(manifest: Dict[str, Dict[str, object]], doc_ids: Sequence[str]) -> None:
    now = now_iso8601()
    for did in doc_ids:
        did_s = str(did).strip()
        if not did_s:
            continue
        cur = manifest.get(did_s) if isinstance(manifest.get(did_s), dict) else {}
        cur = dict(cur or {})
        cur["doc_id"] = did_s
        cur["force_check_next"] = True
        cur["marked_force_at"] = now
        manifest[did_s] = cur


def should_force_check(manifest: Dict[str, Dict[str, object]], doc_id: str) -> bool:
    cur = manifest.get(str(doc_id)) if isinstance(manifest.get(str(doc_id)), dict) else None
    if not cur:
        return False
    return bool(cur.get("force_check_next"))


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(Path(db_path).resolve()))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_documents_table(con: sqlite3.Connection) -> None:
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
    con.execute("CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_documents_openalex_id ON documents(openalex_id);")
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


def iter_doc_ids_from_fts(con: sqlite3.Connection) -> List[str]:
    out: List[str] = []
    try:
        rows = con.execute("SELECT DISTINCT doc_id FROM chunks_fts WHERE doc_id IS NOT NULL AND doc_id != '';").fetchall()
        out = [str(r["doc_id"]) for r in rows if r and r["doc_id"]]
    except Exception as e:
        logger.warning("Failed to list doc_ids from chunks_fts | %s", e)
    return sorted({d for d in out if d})


def seed_documents(con: sqlite3.Connection, doc_ids: Sequence[str]) -> None:
    if not doc_ids:
        return
    con.executemany("INSERT OR IGNORE INTO documents(doc_id) VALUES (?);", [(str(d),) for d in doc_ids if str(d)])


def select_pending_docs(con: sqlite3.Connection, *, limit: Optional[int]) -> List[str]:
    sql = """
        SELECT doc_id
        FROM documents
        WHERE doi IS NULL OR doi = '' OR openalex_id IS NULL OR openalex_id = ''
           OR openalex_title IS NULL OR openalex_title = ''
           OR openalex_authors_json IS NULL OR openalex_authors_json = ''
        ORDER BY doc_id ASC
    """
    rows = con.execute(sql + (" LIMIT ?" if limit is not None else ""), ((int(limit),) if limit is not None else ())).fetchall()  # type: ignore[arg-type]
    return [str(r["doc_id"]) for r in rows if r and r["doc_id"]]


def get_source_md_for_doc(con: sqlite3.Connection, doc_id: str) -> Optional[Path]:
    try:
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
        if not p:
            return None
        return Path(p)
    except Exception as e:
        logger.warning("Failed to get source_md for doc_id=%s | %s", doc_id, e)
        return None


@dataclass(frozen=True)
class EnrichResult:
    doc_id: str
    doi: Optional[str]
    openalex_id: Optional[str]
    updated: bool


def update_documents_row(con: sqlite3.Connection, *, doc_id: str, doi: str, oa: Dict[str, object]) -> None:
    authors_json = None
    oa_authors = oa.get("authors")
    if isinstance(oa_authors, list):
        authors_clean = [str(a).strip() for a in oa_authors if isinstance(a, str) and str(a).strip()]
        if authors_clean:
            authors_json = json.dumps(authors_clean, ensure_ascii=False)

    con.execute(
        """
        UPDATE documents
        SET
            doi = ?,
            openalex_id = ?,
            citation_count = ?,
            publication_year = ?,
            journal_name = ?,
            openalex_title = ?,
            openalex_authors_json = ?,
            updated_at = ?
        WHERE doc_id = ?;
        """,
        (
            str(doi),
            str(oa.get("id") or ""),
            oa.get("cited_by_count"),
            oa.get("publication_year"),
            str(oa.get("journal_name") or "") if oa.get("journal_name") is not None else None,
            str(oa.get("title") or "") if oa.get("title") is not None else None,
            authors_json,
            now_iso8601(),
            str(doc_id),
        ),
    )


def merge_openalex_overwrite_frontmatter(
    fm: Dict[str, object],
    *,
    doc_id: str,
    short_name: str,
    doi: str,
    oa: Dict[str, object],
) -> Dict[str, object]:
    out: Dict[str, object] = dict(fm) if isinstance(fm, dict) else {}
    did = str(doc_id or "").strip()
    sn = str(short_name or "").strip()
    if did:
        out["doc_id"] = did
    if sn:
        out["short_name"] = sn
    out["doi"] = str(doi).strip()
    oid = oa.get("id")
    if oid:
        out["openalex_id"] = str(oid).strip()
    t = oa.get("title")
    if isinstance(t, str) and t.strip():
        out["title"] = t.strip()
    if "authors" in oa:
        au = oa.get("authors")
        if isinstance(au, list):
            out["authors"] = [str(a).strip() for a in au if isinstance(a, str) and str(a).strip()]
        else:
            out["authors"] = []
    if "publication_year" in oa and oa.get("publication_year") is not None:
        try:
            out["year"] = int(oa["publication_year"])  # type: ignore[arg-type]
        except Exception:
            pass
    if "journal_name" in oa:
        out["venue"] = str(oa.get("journal_name") or "").strip()
    if "cited_by_count" in oa and oa.get("cited_by_count") is not None:
        try:
            out["citation_count"] = int(oa["cited_by_count"])  # type: ignore[arg-type]
        except Exception:
            out["citation_count"] = oa.get("cited_by_count")
    return out


def write_markdown_frontmatter_overwrite(src_md: Path, doc: MarkdownDoc, *, doi: str, oa: Dict[str, object]) -> None:
    raw = Path(src_md).read_text(encoding="utf-8", errors="ignore")
    fm, body = split_frontmatter(raw)
    if not isinstance(fm, dict):
        fm = {}
    merged = merge_openalex_overwrite_frontmatter(
        fm,
        doc_id=str(doc.doc_id or ""),
        short_name=str(doc.short_name or ""),
        doi=str(doi),
        oa=oa,
    )
    fm_yaml = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True, default_flow_style=False).strip()
    b = (body or "").lstrip("\n")
    Path(src_md).write_text(f"---\n{fm_yaml}\n---\n\n{b}\n", encoding="utf-8")


def enrich_one(con: sqlite3.Connection, *, doc_id: str, sleep_s: float, write_md: bool = True) -> EnrichResult:
    src_md = get_source_md_for_doc(con, doc_id)
    if src_md is None or not src_md.exists():
        logger.warning("source_md not found for doc_id=%s | %s", doc_id, src_md)
        return EnrichResult(doc_id=doc_id, doi=None, openalex_id=None, updated=False)

    try:
        doc = load_markdown_doc(src_md)
    except Exception as e:
        logger.warning("Failed to load markdown doc_id=%s | md=%s | %s", doc_id, src_md, e)
        return EnrichResult(doc_id=doc_id, doi=None, openalex_id=None, updated=False)

    pdf_path = ""
    try:
        text_all = src_md.read_text(encoding="utf-8", errors="ignore")
        if text_all.splitlines()[:1] == ["---"]:
            end = None
            lines = text_all.splitlines()
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    end = i
                    break
            if end is not None:
                fm = yaml.safe_load("\n".join(lines[1:end])) or {}
                if isinstance(fm, dict):
                    pdf_path = str(fm.get("source_pdf") or "").strip()
    except Exception:
        pdf_path = ""

    doi = None
    try:
        doi = extract_or_find_doi(pdf_path, doc.body, doc.title)
    except Exception as e:
        logger.warning("DOI extraction failed | doc_id=%s | %s", doc_id, e)
        doi = None

    if not doi:
        return EnrichResult(doc_id=doc_id, doi=None, openalex_id=None, updated=False)

    time.sleep(max(0.0, float(sleep_s)))

    oa = None
    try:
        oa = fetch_openalex_metadata(doi)
    except Exception as e:
        logger.warning("OpenAlex fetch failed | doc_id=%s doi=%s | %s", doc_id, doi, e)
        oa = None

    if not oa:
        return EnrichResult(doc_id=doc_id, doi=str(doi), openalex_id=None, updated=False)

    update_documents_row(con, doc_id=doc_id, doi=str(doi), oa=oa)
    if write_md:
        try:
            write_markdown_frontmatter_overwrite(src_md, doc, doi=str(doi), oa=oa)
        except Exception as e:
            logger.warning("Failed to overwrite Markdown frontmatter | doc_id=%s | md=%s | %s", doc_id, src_md, e)
    return EnrichResult(doc_id=doc_id, doi=str(doi), openalex_id=str(oa.get("id") or ""), updated=True)


def run_enrich_metadata(
    *,
    config_path: Path,
    fts_db: Optional[Path] = None,
    limit: Optional[int] = None,
    sleep_s: float = 0.3,
    manifest_path: Optional[Path] = None,
    force_check: bool = False,
    mark_force_next: Optional[Sequence[str]] = None,
    write_md: bool = True,
) -> Dict[str, int]:
    cfg = load_app_config(Path(config_path))
    use_fts_db = Path(fts_db).resolve() if fts_db is not None else default_fts_db_path(cfg)
    use_fts_db.parent.mkdir(parents=True, exist_ok=True)

    mpath = Path(manifest_path).resolve() if manifest_path is not None else default_manifest_path(cfg.project_root)
    manifest = load_manifest(mpath)

    with connect(use_fts_db) as con:
        ensure_documents_table(con)
        doc_ids = iter_doc_ids_from_fts(con)
        seed_documents(con, doc_ids)

        if mark_force_next:
            manifest_force_set(manifest, list(mark_force_next))
            save_manifest(mpath, manifest)
            logger.info("Marked force-check-next for %s doc_id(s) in manifest=%s", len(list(mark_force_next)), mpath)

        pending = select_pending_docs(con, limit=None)
        if force_check:
            pending = list(doc_ids)
        else:
            forced = [d for d in doc_ids if should_force_check(manifest, d)]
            pending = sorted({*pending, *forced})
        if limit is not None:
            pending = pending[: int(limit)]
        if not pending:
            logger.info("No pending documents to enrich.")
            return {"docs_ok": 0, "docs_updated": 0, "docs_failed": 0}

        ok = 0
        updated = 0
        failed = 0

        total = len(pending)
        for idx, did in enumerate(pending, start=1):
            try:
                logger.info("[enrich] %s/%s doc_id=%s", idx, total, did)
                res = enrich_one(con, doc_id=did, sleep_s=float(sleep_s), write_md=bool(write_md))
                ok += 1
                if res.updated:
                    updated += 1
                    logger.info("Enriched doc_id=%s doi=%s openalex_id=%s", res.doc_id, res.doi, res.openalex_id)
                else:
                    logger.info("Skipped doc_id=%s doi=%s openalex_id=%s", res.doc_id, res.doi, res.openalex_id)

                m = manifest.get(did) if isinstance(manifest.get(did), dict) else {}
                m = dict(m or {})
                m["doc_id"] = did
                m["last_checked_at"] = now_iso8601()
                if res.updated:
                    m["last_enriched_at"] = now_iso8601()
                    m["force_check_next"] = False
                manifest[did] = m
            except Exception as e:
                failed += 1
                logger.error("Enrich failed doc_id=%s | %s", did, e)
                continue

        try:
            save_manifest(mpath, manifest)
        except Exception as e:
            logger.warning("Failed to save manifest | path=%s | %s", mpath, e)

        return {"docs_ok": ok, "docs_updated": updated, "docs_failed": failed}

