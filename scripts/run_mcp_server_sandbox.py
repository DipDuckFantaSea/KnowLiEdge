from __future__ import annotations

import argparse
import os
from pathlib import Path

from knotliedge.logging_utils.setup import setup_logging
from knotliedge.config.load import load_app_config
from knotliedge.mcp.server import create_mcp_app

from knotliedge.ops.chroma_sidecar import start_chroma_sidecar


def main() -> None:
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Run KnotLiEdge MCP server (FastMCP).")
    parser.add_argument("--config", type=str, default="sandbox/configs/sandbox.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_app_config(Path(args.config))
    if str(os.environ.get("KNOTLIEDGE_DISABLE_CHROMA_SIDECAR") or "").strip() not in {"1", "true", "TRUE", "yes", "YES"}:
        start_chroma_sidecar(
            db_path=str(cfg.paths.chroma_db_dir),
            host=str(cfg.chroma.http_host),
            port=int(cfg.chroma.http_port),
            log_path=str((cfg.project_root / "output" / f"_chroma_sidecar_{cfg.chroma.http_port}.log").resolve()),
        )

    mcp = create_mcp_app(config_path=Path(args.config))
    logger.info("Starting MCP server with config: %s", args.config)
    mcp.run()


if __name__ == "__main__":
    main()

