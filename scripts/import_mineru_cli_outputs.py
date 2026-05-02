from __future__ import annotations

import argparse
from pathlib import Path

from knotliedge.config.load import load_app_config
from knotliedge.config.paths import ensure_dirs
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.pipeline.import_mineru_cli import run_import_mineru_cli_outputs


def main() -> None:
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Import MinerU CLI output directory into project markdown_vault/assets layout.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing MinerU CLI .md outputs.")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Optional directory containing original PDFs (matched by md filename stem) to compute stable doc_id.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of markdown files to import.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip if target {doc_id}.md already exists.")
    args = parser.parse_args()

    cfg = load_app_config(Path(args.config))
    ensure_dirs(cfg)

    stats = run_import_mineru_cli_outputs(
        cfg,
        input_dir=Path(args.input_dir),
        pdf_dir=Path(args.pdf_dir) if args.pdf_dir else None,
        limit=args.limit,
        skip_existing=not bool(args.no_skip_existing),
    )
    logger.info(
        "Done. total_md=%s succeeded=%s skipped=%s failed=%s",
        stats.total_md,
        stats.succeeded,
        stats.skipped,
        stats.failed,
    )


if __name__ == "__main__":
    main()

