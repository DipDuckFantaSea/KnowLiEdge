from __future__ import annotations

import argparse
import datetime as dt
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.metadata.local_vault_presence import build_radar_local_match_map
from knotliedge.storage.venue_radar_store import VenueRadarStore, default_venue_radar_db_path

logger = logging.getLogger(__name__)


def _read_radar_minimal(db_path: Path, *, limit: Optional[int] = None) -> List[Dict[str, str]]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        sql = "SELECT id, openalex_id, doi, title FROM venue_abstracts"
        params: List[object] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows: List[Dict[str, str]] = []
        for r in con.execute(sql, params):
            rows.append(
                {
                    "id": str(r["id"] or ""),
                    "openalex_id": str(r["openalex_id"] or ""),
                    "doi": str(r["doi"] or ""),
                    "title": str(r["title"] or ""),
                }
            )
        return rows
    finally:
        try:
            con.close()
        except Exception:
            pass


def mark_radar_local_presence(*, config_path: Path, limit: Optional[int] = None) -> int:
    cfg = load_app_config(Path(config_path))
    radar_db = VenueRadarStore(db_path=default_venue_radar_db_path(cfg))
    db_path = radar_db.db_path

    rows = _read_radar_minimal(db_path, limit=limit)
    if not rows:
        logger.info("No radar rows to mark | sqlite=%s", db_path)
        return 0

    ids = [r["id"] for r in rows if r.get("id")]
    radar_openalex_ids = {r["id"]: (r.get("openalex_id") or r["id"]) for r in rows if r.get("id")}
    radar_dois = {r["id"]: (r.get("doi") or "") for r in rows if r.get("id")}
    radar_titles = {r["id"]: (r.get("title") or "") for r in rows if r.get("id")}

    marked_at = dt.datetime.now(dt.timezone.utc).isoformat()
    matches = build_radar_local_match_map(
        cfg=cfg,
        radar_ids=ids,
        radar_openalex_ids=radar_openalex_ids,
        radar_dois=radar_dois,
        radar_titles=radar_titles,
    )
    updated = radar_db.mark_local_presence(matches=matches, marked_at=marked_at)
    logger.info("Marked local presence | scanned=%s matched=%s updated=%s sqlite=%s", len(ids), len(matches), updated, db_path)
    return int(updated)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Mark venue radar rows with whether they exist in local Markdown vault.")
    parser.add_argument("--config", type=str, default="sandbox/configs/sandbox.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max radar rows to scan.")
    args = parser.parse_args()

    n = mark_radar_local_presence(config_path=Path(args.config), limit=args.limit)
    logger.info("Done. updated=%s", int(n))


if __name__ == "__main__":
    main()

