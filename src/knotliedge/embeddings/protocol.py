from __future__ import annotations

from typing import List, Protocol, Sequence


class Embedder(Protocol):
    """Embedding interface used across the project."""

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts."""

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""

