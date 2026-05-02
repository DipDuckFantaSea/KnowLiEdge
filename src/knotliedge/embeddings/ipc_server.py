from __future__ import annotations

import atexit
import logging
import os
import signal
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Dict, Tuple

from knotliedge.config.load import load_app_config
from knotliedge.embeddings.bge_m3 import EmbeddingModelNotReadyError
from knotliedge.embeddings.lazy_bge_model import LazyBgeM3Model

logger = logging.getLogger(__name__)


def serve_embedding_ipc(*, config_path: Path, host: str, port: int) -> None:
    """Run a single-process embedding server (loads BGE-M3 once, serves many clients).

    Args:
        config_path: App config YAML path.
        host: Host to bind.
        port: Port to bind.

    Returns:
        None. This is a blocking call.
    """
    # Logging: configure only from ``scripts/run_embedding_server`` (single RichHandler).
    # This socket speaks the stdlib multiprocessing ``Client`` wire protocol, not HTTP.
    cfg = load_app_config(Path(config_path))
    lazy_model = LazyBgeM3Model(embedding_cfg=cfg.embedding)

    # Idle auto-unload settings (override via env for testing).
    idle_unload_s = float(os.environ.get("KNOTLIEDGE_EMBED_IDLE_UNLOAD_S", "600"))
    check_interval_s = float(os.environ.get("KNOTLIEDGE_EMBED_IDLE_CHECK_S", "60"))
    lazy_model.start_idle_unload_guard(idle_unload_s=idle_unload_s, check_interval_s=check_interval_s)
    atexit.register(lambda: lazy_model.unload(reason="atexit"))

    addr: Tuple[str, int] = (str(host), int(port))
    listener = Listener(addr, authkey=b"knotliedge")
    listener_ref: Dict[str, Listener] = {"l": listener}

    def _close_listener(*_args: object) -> None:
        """Unblock ``accept()`` so Ctrl+C / SIGBREAK can exit on Windows."""

        try:
            listener_ref["l"].close()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _close_listener)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _close_listener)

    logger.info(
        "Embedding IPC listening %s:%s (multiprocessing Client auth; not HTTP). "
        "BGE-M3 loads on first embed request. Ctrl+C closes listener.",
        host,
        port,
    )

    try:
        while True:
            try:
                conn = listener.accept()
            except (OSError, ConnectionError, ValueError, EOFError) as e:
                # Listener closed (e.g. Ctrl+C) or transient Windows socket errors.
                if "closed" in str(e).lower() or "bad file" in str(e).lower():
                    logger.info("Embedding IPC listener closed: %s", e)
                    break
                logger.warning("IPC accept failed: %s", e)
                continue

            try:
                try:
                    req = conn.recv()
                except (EOFError, OSError, ConnectionError) as e:
                    logger.warning("IPC client disconnected before request: %s", e)
                    continue
                if not isinstance(req, dict):
                    conn.send({"ok": False, "error": "bad_request"})
                    continue
                op = str(req.get("op") or "")
                if op == "embed_texts":
                    texts = req.get("texts") or []
                    if not isinstance(texts, list):
                        conn.send({"ok": False, "error": "texts_not_list"})
                        continue
                    try:
                        embedder = lazy_model.get()
                    except EmbeddingModelNotReadyError as e:
                        conn.send({"ok": False, "error": str(e)})
                        continue
                    vecs = embedder.embed_texts([str(t) for t in texts])
                    conn.send({"ok": True, "vectors": vecs})
                elif op == "embed_query":
                    q = str(req.get("query") or "")
                    try:
                        embedder = lazy_model.get()
                    except EmbeddingModelNotReadyError as e:
                        conn.send({"ok": False, "error": str(e)})
                        continue
                    v = embedder.embed_query(q)
                    conn.send({"ok": True, "vector": v})
                else:
                    conn.send({"ok": False, "error": f"unknown_op:{op}"})
            except Exception as e:
                try:
                    conn.send({"ok": False, "error": str(e)})
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: closing embedding IPC server")
    finally:
        _close_listener()
        try:
            lazy_model.unload(reason="server_shutdown")
        except Exception:
            pass

