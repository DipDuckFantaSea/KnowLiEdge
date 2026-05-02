"""Embedding models."""

from knotliedge.embeddings.bge_m3 import BgeM3Embedder, EmbeddingModelNotReadyError
from knotliedge.embeddings.factory import get_embedder
from knotliedge.embeddings.lazy_ipc_gate import LazyIpcEmbedderSession
from knotliedge.embeddings.protocol import Embedder

__all__ = [
    "BgeM3Embedder",
    "EmbeddingModelNotReadyError",
    "Embedder",
    "get_embedder",
    "LazyIpcEmbedderSession",
]

