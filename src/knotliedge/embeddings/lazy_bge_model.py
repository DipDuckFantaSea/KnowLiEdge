"""Lazy singleton for the in-process BGE-M3 model (IPC embedding server only).

``scripts.run_embedding_server`` should bind its listener before loading
SentenceTransformer weights so the port is reachable while VRAM is still free.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Optional

from knotliedge.config.types import EmbeddingConfig
from knotliedge.embeddings.bge_m3 import BgeM3Embedder

# Do not call setup_logging() here: the embedding server entrypoint configures the
# ``knotliedge`` tree once; multiple RichHandlers + background threads garble Windows consoles.
logger = logging.getLogger(__name__)


class LazyBgeM3Model:
    """Load ``BgeM3Embedder`` on first encode call (one config per server process)."""

    def __init__(self, *, embedding_cfg: EmbeddingConfig) -> None:
        self._embedding_cfg = embedding_cfg
        self._model: Optional[BgeM3Embedder] = None
        self._last_used_time: float = 0.0
        self._lock = threading.Lock()
        self._started_guard: bool = False

    def touch(self) -> None:
        """Update last-used time for idle unload logic."""
        with self._lock:
            self._last_used_time = float(time.time())

    def start_idle_unload_guard(
        self,
        *,
        idle_unload_s: float = 600.0,
        check_interval_s: float = 60.0,
    ) -> None:
        """Start a daemon thread to auto-unload model after idle period.

        Args:
            idle_unload_s: Idle seconds after which the model is unloaded.
            check_interval_s: Period to check idle state.

        Returns:
            None.
        """
        with self._lock:
            if self._started_guard:
                return
            self._started_guard = True

        def _loop() -> None:
            logger.info(
                "Embedding idle auto-unload guard started (idle_unload_s=%s check_interval_s=%s)",
                float(idle_unload_s),
                float(check_interval_s),
            )
            while True:
                try:
                    time.sleep(float(check_interval_s))
                    now = float(time.time())
                    with self._lock:
                        m = self._model
                        last = float(self._last_used_time)
                    if m is None:
                        continue
                    if last <= 0.0:
                        continue
                    if now - last < float(idle_unload_s):
                        continue
                    self.unload(reason="idle_timeout")
                except Exception as e:
                    logger.warning("Idle auto-unload guard loop error: %s", e)

        t = threading.Thread(
            target=_loop,
            name="knotliedge-embedding-idle-unload",
            daemon=True,
        )
        t.start()

    def get(self) -> BgeM3Embedder:
        """Return the embedder, constructing it on first use.

        Returns:
            A shared ``BgeM3Embedder`` instance.

        Raises:
            EmbeddingModelNotReadyError: If model construction fails.
        """
        with self._lock:
            if self._model is None:
                logger.info("Lazy loading BGE-M3 (SentenceTransformer) in embedding IPC server...")
                self._model = BgeM3Embedder(self._embedding_cfg)
            self._last_used_time = float(time.time())
            return self._model

    def unload(self, *, reason: str) -> None:
        """Unload the in-process model and try to release GPU memory.

        Args:
            reason: Human-readable reason for unload (e.g. 'idle_timeout', 'atexit').

        Returns:
            None.
        """
        t0 = time.time()
        with self._lock:
            if self._model is None:
                return
            self._model = None
            self._last_used_time = float(time.time())

        # Force Python GC first.
        gc.collect()

        # Best-effort torch CUDA cache cleanup (do not crash if torch unavailable).
        try:
            import torch  # type: ignore

            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        dt_ms = int((time.time() - t0) * 1000.0)
        logger.info("Unloaded embedding model (reason=%s, dt_ms=%s)", str(reason), dt_ms)
