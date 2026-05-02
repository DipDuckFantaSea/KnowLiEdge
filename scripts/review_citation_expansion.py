from __future__ import annotations

import argparse
from pathlib import Path

from knotliedge.citation_graph.hitl import OpenAlexExpansionStagingStore
from knotliedge.citation_graph.store import default_citation_db_path
from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Review/approve staged OpenAlex citation expansion candidates (HITL).")
    ap.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    ap.add_argument("--db", type=str, default=None, help="Override citations.sqlite3 path.")
    ap.add_argument("--run-id", type=str, required=True, help="Staging run id used by --stage-only expansion.")
    ap.add_argument("--approve-all", action="store_true", help="Approve and materialize all pending items (up to --limit).")
    ap.add_argument("--limit", type=int, default=5000, help="Max staged rows to approve in one run.")
    args = ap.parse_args()

    cfg = load_app_config(Path(args.config))
    db_path = Path(args.db) if args.db else default_citation_db_path(cfg)
    stg = OpenAlexExpansionStagingStore(db_path=db_path)

    c = stg.counts(run_id=str(args.run_id))
    print(
        "\n".join(
            [
                f"run_id: {c.run_id}",
                f"pending_works: {c.pending_works}  pending_edges: {c.pending_edges}",
                f"approved_works: {c.approved_works}  approved_edges: {c.approved_edges}",
                f"rejected_works: {c.rejected_works}  rejected_edges: {c.rejected_edges}",
                f"db: {db_path.resolve()}",
            ]
        )
    )

    if args.approve_all:
        res = stg.approve_all_pending(run_id=str(args.run_id), limit=int(args.limit))
        print(f"approved_works: {res.get('approved_works')}  approved_edges: {res.get('approved_edges')}")


if __name__ == "__main__":
    main()
    raise SystemExit(0)

