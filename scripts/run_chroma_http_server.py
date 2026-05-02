"""Start standalone Chroma (HTTP) using persist path and port from a KnotLiEdge YAML config.

Requires the ``chroma`` CLI on PATH (typically installed with ``pip install chromadb``).
Example::

    python scripts/run_chroma_http_server.py --config sandbox/configs/sandbox.yaml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from knotliedge.config.load import load_app_config

    ap = argparse.ArgumentParser(description="Run Chroma HTTP server (daemon) for KnotLiEdge.")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML (default or sandbox).")
    args = ap.parse_args()

    cfg = load_app_config(Path(args.config).resolve())
    persist = str(cfg.paths.chroma_db_dir)
    port = int(cfg.chroma.http_port)
    chroma_exe = shutil.which("chroma")
    if not chroma_exe:
        print(
            "未在 PATH 中找到 `chroma` 命令。请安装 chromadb 后重试，或用手动命令启动：\n"
            f"  chroma run --path \"{persist}\" --port {port}\n"
            "（端口须与 yaml 中 chroma.http_port 一致；持久化目录须与此配置的 chroma_db_dir 一致。）",
            file=sys.stderr,
        )
        return 1

    cmd = [chroma_exe, "run", "--path", persist, "--port", str(port)]
    print("exec:", " ".join(cmd), flush=True)
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
