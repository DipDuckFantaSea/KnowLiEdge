from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from knotliedge.config.load import resolve_project_root_for_config
from knotliedge.embeddings.protocol import Embedder

logger = logging.getLogger(__name__)


def _default_addr() -> Tuple[str, int]:
    # A fixed localhost port for the single-model embedding service.
    return ("127.0.0.1", 60123)


def _can_connect(host: str, port: int, *, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
            return True
    except Exception:
        return False


def _start_server_subprocess(*, config_path: Path, host: str, port: int) -> None:
    """Best-effort start embedding server in background."""
    # Use current interpreter (should already be conda env `agent`).
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_embedding_server",
        "--config",
        str(config_path),
        "--host",
        str(host),
        "--port",
        str(int(port)),
    ]
    try:
        repo_root = resolve_project_root_for_config(config_path)
        subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    except Exception as e:
        logger.warning("Failed to start embedding server subprocess: %s", e)


@dataclass(frozen=True)
class IpcEmbedder(Embedder):
    """Embedder client backed by a single local embedding server process."""

    host: str
    port: int
    timeout_s: float = 120.0

    def _call(self, payload: Dict[str, Any]) -> Any:
        conn = Client((self.host, int(self.port)), authkey=b"knotliedge")
        try:
            conn.send(payload)
            return conn.recv()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        res = self._call({"op": "embed_texts", "texts": list(texts)})
        if not isinstance(res, dict) or not res.get("ok"):
            raise RuntimeError(f"embedding server error: {res}")
        vecs = res.get("vectors") or []
        return [list(v) for v in vecs]

    def embed_query(self, query: str) -> List[float]:
        res = self._call({"op": "embed_query", "query": str(query)})
        if not isinstance(res, dict) or not res.get("ok"):
            raise RuntimeError(f"embedding server error: {res}")
        v = res.get("vector") or []
        return list(v)


def get_ipc_embedder(
    *,
    config_path: Path,
    host: Optional[str] = None,
    port: Optional[int] = None,
    autostart: bool = True,
    wait_s: float = 20.0,
) -> IpcEmbedder:
    """Get an IPC embedder client, optionally auto-starting the server.

    Args:
        config_path: Path to app YAML config (used for auto-start).
        host: Server host (default: 127.0.0.1).
        port: Server port (default: 60123).
        autostart: If true, start the server if not reachable.
        wait_s: Max seconds to wait for server readiness after start.

    Returns:
        IpcEmbedder client.
    """
    h, p0 = _default_addr()
    h = str(host or h)
    p = int(port or p0)
    if _can_connect(h, p, timeout_s=0.2):
        return IpcEmbedder(host=h, port=p)

    if autostart:
        _start_server_subprocess(config_path=config_path, host=h, port=p)
        t0 = time.time()
        while time.time() - t0 < float(wait_s):
            if _can_connect(h, p, timeout_s=0.2):
                return IpcEmbedder(host=h, port=p)
            time.sleep(0.2)

    raise RuntimeError(f"Embedding IPC server not reachable at {h}:{p}")

