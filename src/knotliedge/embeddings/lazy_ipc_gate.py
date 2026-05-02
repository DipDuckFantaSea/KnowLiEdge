"""Defer obtaining the IPC embedding client until first vector operation.

The MCP process must not call ``get_embedder`` at import or ``create_mcp_app``
time: doing so can auto-start ``scripts.run_embedding_server`` and allocate GPU
memory before any tool needs embeddings (e.g. leaving VRAM for BERTopic).
"""

from __future__ import annotations

import atexit
import gc
import logging
import sys
from pathlib import Path
from typing import Optional

from knotliedge.embeddings.factory import get_embedder
from knotliedge.embeddings.protocol import Embedder
from knotliedge.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


_ATEEXIT_GPU_CLEANUP_REGISTERED: bool = False


def _cleanup_gpu_memory_best_effort() -> None:
    """Best-effort GPU cleanup for MCP process exit.

    Notes:
        - The MCP process typically does not hold the embedding model (GPU is in IPC server),
          but we keep this as a safety net for unexpected in-process CUDA usage.
        - Do not force-import torch; only act if it is already imported.
    """
    try:
        gc.collect()
    except Exception:
        pass

    if "torch" not in sys.modules:
        return
    try:
        import torch  # type: ignore

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _register_atexit_gpu_cleanup_once() -> None:
    global _ATEEXIT_GPU_CLEANUP_REGISTERED
    if _ATEEXIT_GPU_CLEANUP_REGISTERED:
        return
    atexit.register(_cleanup_gpu_memory_best_effort)
    _ATEEXIT_GPU_CLEANUP_REGISTERED = True


class LazyIpcEmbedderSession:
    """Lazy singleton around ``get_embedder`` for one MCP server lifetime.

    Attributes:
        config_path: YAML path passed to ``get_embedder``.
    """

    def __init__(self, *, config_path: Path) -> None:
        _register_atexit_gpu_cleanup_once()
        self._config_path = Path(config_path).resolve()
        self._embedder: Optional[Embedder] = None
        self._failed: bool = False

    @property
    def ready(self) -> bool:
        """True only after a successful lazy init (does not trigger load)."""
        return self._embedder is not None

    def invalidate(self) -> None:
        """Drop cached embedder so the next call reconnects (and may autostart IPC server)."""

        self._embedder = None
        self._failed = False

    def get_optional(self) -> Optional[Embedder]:
        """Return the IPC embedder, attempting init once; ``None`` on failure.

        Returns:
            An ``Embedder`` instance, or ``None`` if initialization failed.
        """
        if self._failed:
            return None
        if self._embedder is not None:
            return self._embedder
        try:
            self._embedder = get_embedder(config_path=self._config_path)
            return self._embedder
        except Exception as e:
            logger.error("Lazy IPC embedder init failed: %s", e)
            self._failed = True
            return None

    def require_for_store(self, store: ChromaStore) -> Embedder:
        """Resolve embedder, bind it to ``store``, or raise with a clear message.

        Args:
            store: Chroma store that will run ``search`` / smoke queries.

        Returns:
            A ready ``Embedder``.

        Raises:
            RuntimeError: If embedding is unavailable after the first init attempt.
        """
        e = self.get_optional()
        if e is None:
            raise RuntimeError(
                "Embedding/Chroma 尚未就绪。需要你手动安装 torch + 准备本地模型权重，并先运行 index_markdown 入库。"
            )
        store.bind_embedder(e)
        return e
