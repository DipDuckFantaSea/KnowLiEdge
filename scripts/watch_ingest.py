from __future__ import annotations

import argparse
import math
from pathlib import Path

from knotliedge.config.load import load_app_config
from knotliedge.config.paths import ensure_dirs
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.pipeline.watch_ingest import run_watch_ingest


def main() -> None:
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Watch raw PDFs and ingest into vault + ChromaDB.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--poll", type=float, default=3.0, help="Polling interval seconds.")
    parser.add_argument("--max-loops", type=int, default=None, help="Stop after N polling loops (for tests).")
    parser.add_argument("--api-url", type=str, default=None, help="MinerU api url. If omitted, start local mineru-api.")
    parser.add_argument("--enable-vlm-preload", action="store_true", help="Enable VLM preload when starting mineru-api.")
    # Safe defaults: avoid VLM engines unless user explicitly opts in.
    parser.add_argument("--backend", type=str, default="pipeline", help="MinerU backend (e.g., pipeline, hybrid-auto-engine).")
    parser.add_argument("--method", type=str, default="txt", help="MinerU parse method (auto/txt/ocr).")
    parser.add_argument("--no-formula", action="store_true", help="Disable formula parsing.")
    parser.add_argument("--no-table", action="store_true", help="Disable table parsing.")
    parser.add_argument("--export-images", action="store_true", help="Export images into vault/images (requires zip result).")
    parser.add_argument("--max-docs", type=int, default=None, help="Stop after attempting N new docs (ok/failed).")
    parser.add_argument(
        "--parse-mode",
        type=str,
        default="api",
        help="PDF parsing mode. 目前已强制为 api（mineru-api）。",
    )
    parser.add_argument(
        "--mineru-model-source",
        type=str,
        default="local",
        help="Set env MINERU_MODEL_SOURCE for mineru-api (recommended: local).",
    )
    parser.add_argument(
        "--mineru-virtual-vram-size",
        type=int,
        default=8,
        help="Set env MINERU_VIRTUAL_VRAM_SIZE for mineru-api (likely GB; e.g., 8).",
    )
    args = parser.parse_args()

    cfg = load_app_config(Path(args.config))
    ensure_dirs(cfg)

    stats = run_watch_ingest(
        cfg,
        config_path=Path(args.config),
        api_url=str(args.api_url) if args.api_url else None,
        enable_vlm_preload=bool(args.enable_vlm_preload),
        poll_interval_s=float(args.poll),
        max_loops=args.max_loops,
        max_docs=args.max_docs,
        parse_mode=str(args.parse_mode),
        mineru_model_source=str(args.mineru_model_source),
        mineru_virtual_vram_size=int(args.mineru_virtual_vram_size) if args.mineru_virtual_vram_size is not None else None,
        parse_backend=str(args.backend),
        parse_method=str(args.method),
        formula_enable=not bool(args.no_formula),
        table_enable=not bool(args.no_table),
        export_images=bool(args.export_images),
    )
    logger.info(
        "Watch done. seen_total=%s ok=%s skipped=%s failed=%s",
        stats.seen_total,
        stats.processed_ok,
        stats.processed_skipped,
        stats.processed_failed,
    )
    if getattr(stats, "parse_seconds_ok", None):
        s = sorted(float(x) for x in stats.parse_seconds_ok if isinstance(x, (int, float)))
        if s:
            mean = sum(s) / len(s)
            p50 = s[int(math.floor(0.50 * (len(s) - 1)))]
            p90 = s[int(math.floor(0.90 * (len(s) - 1)))]
            logger.info("Parse seconds summary: n=%s mean=%.3f p50=%.3f p90=%.3f max=%.3f", len(s), mean, p50, p90, s[-1])


if __name__ == "__main__":
    main()

