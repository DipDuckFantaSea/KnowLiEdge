from __future__ import annotations

import argparse
from pathlib import Path

from knotliedge.config.load import load_app_config
from knotliedge.config.paths import ensure_dirs
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.pipeline.pdf_to_md import run_pdf_to_md


def main() -> None:
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="PDF -> Markdown vault (MinerU).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process.")
    args = parser.parse_args()

    cfg = load_app_config(Path(args.config))
    ensure_dirs(cfg)

    stats = run_pdf_to_md(cfg, limit=args.limit)
    logger.info("Done. total=%s succeeded=%s skipped=%s failed=%s", stats.total, stats.succeeded, stats.skipped, stats.failed)


if __name__ == "__main__":
    main()

