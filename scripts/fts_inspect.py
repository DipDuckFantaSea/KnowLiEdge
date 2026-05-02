from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect KnotLiEdge FTS sqlite schema.")
    p.add_argument("--db", required=True, help="Path to fts.sqlite3")
    args = p.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"FTS db not found: {db}")

    con = sqlite3.connect(str(db))
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("select name from sqlite_master where type='table' order by name").fetchall()]
    print("tables:")
    for t in tables:
        print("-", t)
        cols = cur.execute(f"pragma table_info({t})").fetchall()
        for _, name, ctype, *_ in cols:
            print(f"    {name} {ctype}")


if __name__ == "__main__":
    main()

