from __future__ import annotations

import argparse
from pathlib import Path

from knotliedge.embeddings.ipc_server import serve_embedding_ipc
from knotliedge.logging_utils.setup import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run a single-process local embedding IPC server (BGE-M3).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=60123, help="Bind port.")
    args = parser.parse_args()
    serve_embedding_ipc(config_path=Path(args.config), host=str(args.host), port=int(args.port))


if __name__ == "__main__":
    main()

