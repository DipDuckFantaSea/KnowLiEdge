from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from knotliedge.config.types import EmbeddingConfig
from knotliedge.logging_utils.setup import setup_logging

logger = setup_logging()


class EmbeddingModelNotReadyError(RuntimeError):
    """Raised when embedding backend/model weights are not ready."""


@dataclass
class BgeM3Embedder:
    """A thin wrapper around SentenceTransformer for BGE-M3 style models."""

    cfg: EmbeddingConfig

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise EmbeddingModelNotReadyError(
                "Embedding 依赖导入失败（sentence-transformers/torch）。通常是 torch 安装不匹配导致 DLL 加载失败。"
                "需要你手动安装匹配的 torch（CPU/GPU），并确保 `import torch` 能成功。"
            ) from e

        try:
            self._model = SentenceTransformer(self.cfg.model_name_or_path, device=self.cfg.device)
        except Exception as e:
            raise EmbeddingModelNotReadyError(
                "Embedding 模型加载失败。需要你手动准备本地模型权重，并在 configs/default.yaml 中正确设置 "
                "`embedding.model_name_or_path`。此外还需要你手动安装匹配的 torch（CPU/GPU）。"
            ) from e

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts.

        Args:
            texts: Input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        vecs = self._model.encode(
            list(texts),
            batch_size=int(self.cfg.batch_size),
            normalize_embeddings=bool(self.cfg.normalize_embeddings),
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

