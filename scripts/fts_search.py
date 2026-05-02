from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Search KnotLiEdge FTS (chunks_fts) for keywords.")
    p.add_argument("--db", required=True, help="Path to fts.sqlite3")
    p.add_argument("--query", required=True, help="FTS query string (e.g. 'GaN-on-Si' or 'HEMT')")
    p.add_argument("--limit", type=int, default=50, help="Max rows to return")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"FTS db not found: {db}")

    con = sqlite3.connect(str(db))
    cur = con.cursor()

    # Expect schema:
    # - chunks_fts(chunk_id, doc_id, short_name, section, text)
    # - chunks_meta(chunk_id, source_md, chunk_index, created_at)
    rows = cur.execute(
        "select f.chunk_id, f.doc_id, f.short_name, f.section, m.source_md, substr(f.text, 1, 220) as snippet "
        "from chunks_fts f left join chunks_meta m on m.chunk_id = f.chunk_id "
        "where chunks_fts match ? limit ?",
        (args.query, int(args.limit)),
    ).fetchall()

    def safe_print(line: str) -> None:
        # PowerShell defaults to legacy encodings; force utf-8 bytes.
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))

    safe_print(f"db={db}")
    safe_print(f"query={args.query!r} limit={args.limit} hits={len(rows)}")
    for i, (chunk_id, doc_id, short_name, section, source_md, snippet) in enumerate(rows, start=1):
        s = (snippet or "").replace("\n", " ").replace("\r", " ").strip()
        sec = (section or "").replace("\n", " ").strip()
        sn = (short_name or "").replace("\n", " ").strip()
        smd = (source_md or "").replace("\n", " ").replace("\r", " ").strip()
        safe_print(
            f"{i:03d} doc_id={doc_id} chunk_id={chunk_id} short_name={sn} section={sec} "
            f"source_md={smd} snippet={s}"
        )


if __name__ == "__main__":
    main()

