from __future__ import annotations

import argparse
import logging
from pathlib import Path

from knotliedge.logging_utils.setup import setup_logging
from knotliedge.venue_radar.radar import run_venue_radar

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run Parallel Venue Radar: fetch OpenAlex abstracts into quarantine stores.")
    parser.add_argument("--config", type=str, default="sandbox/configs/sandbox.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max works to ingest (across venues).")
    parser.add_argument("--lookback_days", type=int, default=None, help="Override venue_radar.lookback_days (days).")
    parser.add_argument("--no_fetch", action="store_true", help="Only initialize stores (no network fetch).")
    parser.add_argument("--purge", action="store_true", help="Purge radar Chroma/SQLite before fetching.")
    parser.add_argument("--no_mark_local", action="store_true", help="Skip scanning local vault and marking radar rows.")
    args = parser.parse_args()

    n = run_venue_radar(
        config_path=Path(args.config),
        limit=args.limit,
        no_fetch=bool(args.no_fetch),
        purge=bool(args.purge),
        mark_local=not bool(args.no_mark_local),
        lookback_days=(int(args.lookback_days) if args.lookback_days is not None else None),
    )
    logger.info("Venue radar done. ingested=%s", int(n))


if __name__ == "__main__":
    main()

