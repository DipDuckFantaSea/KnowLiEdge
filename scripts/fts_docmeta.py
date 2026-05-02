from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch doc metadata from KnotLiEdge FTS sqlite.")
    p.add_argument("--db", required=True, help="Path to fts.sqlite3")
    p.add_argument("--doc-id", action="append", default=[], help="Doc id (repeatable)")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"FTS db not found: {db}")

    doc_ids = [d.strip() for d in (args.doc_id or []) if d and d.strip()]
    if not doc_ids:
        raise SystemExit("Provide at least one --doc-id")

    con = sqlite3.connect(str(db))
    cur = con.cursor()

    placeholders = ",".join(["?"] * len(doc_ids))
    rows = cur.execute(
        f"select doc_id, publication_year, journal_name, doi, openalex_id, openalex_title, citation_count "
        f"from documents where doc_id in ({placeholders})",
        tuple(doc_ids),
    ).fetchall()

    rows_by_id = {r[0]: r for r in rows}
    for did in doc_ids:
        r = rows_by_id.get(did)
        if not r:
            print(f"doc_id={did} NOT_FOUND")
            continue
        _, year, journal, doi, oa_id, title, cites = r
        print(
            f"doc_id={did} year={year} cites={cites} doi={doi} openalex_id={oa_id} journal={journal} title={title}"
        )


if __name__ == "__main__":
    main()

