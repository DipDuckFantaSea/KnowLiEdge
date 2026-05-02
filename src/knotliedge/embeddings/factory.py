from __future__ import annotations

from pathlib import Path

from knotliedge.embeddings.ipc import get_ipc_embedder
from knotliedge.embeddings.protocol import Embedder


def get_embedder(*, config_path: Path) -> Embedder:
    """Return the project-wide embedder.

    Strategy:
    - Prefer the local IPC embedding server (single-model, queued).
    - Auto-start the server if missing.

    Args:
        config_path: YAML config path for auto-start.

    Returns:
        An object implementing ``Embedder``.
    """
    return get_ipc_embedder(config_path=Path(config_path), autostart=True)

