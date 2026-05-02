from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.storage.fts_store import default_fts_db_path

logger = logging.getLogger(__name__)


def _read_frontmatter_title(md_path: Path) -> Optional[Dict[str, str]]:
    try:
        raw = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    if not raw.startswith("---"):
        return None
    end = raw.find("\n---", 3)
    if end == -1:
        return None
    block = raw[3:end]
    try:
        fm = yaml.safe_load(block) or {}
    except Exception:
        return None
    if not isinstance(fm, dict):
        return None
    did = str(fm.get("doc_id") or "").strip()
    title = str(fm.get("title") or "").strip()
    if not did or not title:
        return None
    return {"doc_id": did, "title": title}


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Detect mismatches between vault frontmatter title and OpenAlex title.")
    ap.add_argument("--config", type=str, default="sandbox/configs/sandbox.yaml", help="Config path.")
    ap.add_argument("--fts-db", type=str, default=None, help="Override fts.sqlite3 path.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of mismatches to output.")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output json path (default: output/_title_mismatches_<env>.json under project root).",
    )
    args = ap.parse_args()

    cfg = load_app_config(Path(args.config))
    vault_dir = Path(cfg.paths.markdown_vault_dir).resolve()
    fts_db = Path(args.fts_db).resolve() if args.fts_db else default_fts_db_path(cfg)

    titles_by_doc: Dict[str, str] = {}
    for md in vault_dir.glob("*.md"):
        got = _read_frontmatter_title(md)
        if not got:
            continue
        titles_by_doc[got["doc_id"]] = got["title"]

    if not fts_db.exists():
        raise FileNotFoundError(str(fts_db))

    with _connect(fts_db) as con:
        rows = con.execute(
            """
            SELECT doc_id, openalex_title
            FROM documents
            WHERE openalex_title IS NOT NULL AND trim(openalex_title) != '';
            """
        ).fetchall()

    mismatches: List[Dict[str, str]] = []
    for r in rows:
        did = str(r["doc_id"] or "").strip()
        oa = str(r["openalex_title"] or "").strip()
        vt = str(titles_by_doc.get(did) or "").strip()
        if not did or not oa or not vt:
            continue
        if oa != vt:
            mismatches.append({"doc_id": did, "vault_title": vt, "openalex_title": oa})

    if args.limit is not None:
        mismatches = mismatches[: int(args.limit)]

    out_path = Path(args.out).resolve() if args.out else (Path(cfg.project_root) / "output" / f"_title_mismatches_{'sandbox' if cfg.environment.is_sandbox else 'default'}.json").resolve()  # type: ignore[attr-defined]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mismatches, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("mismatch_count=%s out=%s", len(mismatches), out_path)
    print(len(mismatches))
    print(str(out_path))


if __name__ == "__main__":
    main()
    raise SystemExit(0)

