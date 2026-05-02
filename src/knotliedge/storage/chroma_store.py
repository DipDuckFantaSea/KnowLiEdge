from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import chromadb
from chromadb.api.models.Collection import Collection

from knotliedge.config.types import AppConfig
from knotliedge.logging_utils.setup import setup_logging

setup_logging()


def _ensure_no_proxy_for_localhost(host: str) -> None:
    """Avoid corporate/system HTTP proxy interfering with localhost HttpClient.

    httpx/httpcore will honor HTTP(S)_PROXY unless NO_PROXY includes localhost.
    """

    h = (host or "").strip().lower()
    if h not in {"localhost", "127.0.0.1", "::1"}:
        return
    existing = (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").strip()
    need = {"localhost", "127.0.0.1", "::1"}
    cur = {x.strip() for x in existing.split(",") if x.strip()} if existing else set()
    merged = cur | need
    out = ",".join(sorted(merged))
    os.environ["NO_PROXY"] = out
    os.environ["no_proxy"] = out


def format_chroma_http_client_error(cfg: AppConfig, exc: BaseException) -> str:
    """Human-readable hint when HttpClient cannot reach the Chroma daemon."""

    host = cfg.chroma.http_host
    port = int(cfg.chroma.http_port)
    persist = cfg.paths.chroma_db_dir
    return (
        f"无法连接 Chroma HTTP 服务 ({host}:{port})。\n"
        f"请先在独立进程中启动 Chroma，并使持久化目录与配置一致：\n"
        f"  chroma_db_dir = {persist}\n"
        f"示例（端口需与 yaml 中 chroma.http_port 一致）：\n"
        f"  chroma run --path \"{persist}\" --port {port}\n"
        f"或 Docker: https://docs.trychroma.com/guides/deployment\n"
        f"底层异常: {type(exc).__name__}: {exc}"
    )


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    doc_id: str
    short_name: str
    section: Optional[str]
    preview: str
    source_md: str


@dataclass
class ChunkContext:
    chunk_id: str
    doc_id: str
    short_name: str
    source_md: str
    section: Optional[str]
    text: str
    chunk_index: int


class ChromaStore:
    """ChromaDB retrieval + writes over **HTTP** (standalone server process)."""

    def __init__(
        self,
        *,
        cfg: AppConfig,
        embedder: Optional[object] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        host = str(cfg.chroma.http_host).strip() or "127.0.0.1"
        port = int(cfg.chroma.http_port)
        _ensure_no_proxy_for_localhost(host)
        try:
            self._client = chromadb.HttpClient(host=host, port=port)
        except Exception as e:
            raise RuntimeError(format_chroma_http_client_error(cfg, e)) from e

        use_name = str(collection_name).strip() if collection_name is not None else str(cfg.chroma.collection_name)
        if not use_name:
            raise ValueError("collection_name must be non-empty")
        self._collection_name = use_name
        try:
            self._collection: Collection = self._client.get_or_create_collection(name=use_name)
        except Exception as e:
            raise RuntimeError(
                format_chroma_http_client_error(cfg, e)
                + f"\n(已连上服务端，但打开集合 {use_name!r} 失败；可检查 HTTP 5xx 或集合名。)"
            ) from e

    def bind_embedder(self, embedder: Optional[object]) -> None:
        """Attach or replace the embedder used for ``search`` (e.g. MCP lazy init).

        Args:
            embedder: Object implementing ``embed_query(str) -> list[float]``, or ``None``.
        """
        self._embedder = embedder

    @property
    def collection_name(self) -> str:
        """Chroma collection name."""
        return str(self._collection_name)

    def collection_count(self) -> int:
        """Return number of items in current collection."""
        return int(self._collection.count())

    def get_metadatas_page(self, *, offset: int, limit: int) -> List[Dict[str, Any]]:
        """Fetch a page of chunk metadatas (for aggregation / health checks).

        Args:
            offset: Row offset for pagination.
            limit: Maximum number of records to return.

        Returns:
            List of metadata dicts (same order as Chroma returns for this page).
        """
        got = self._collection.get(
            offset=int(offset),
            limit=int(limit),
            include=["metadatas"],
        )
        metas = got.get("metadatas") or []
        return [m if isinstance(m, dict) else {} for m in metas]

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks that belong to a document.

        Args:
            doc_id: Document id.

        Returns:
            Number of deleted ids (best-effort).
        """
        doc_id_s = str(doc_id)
        got = self._collection.get(where={"doc_id": doc_id_s}, include=[])
        ids = got.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def get_doc_marker(self, doc_id: str) -> Optional[Dict[str, object]]:
        """Get a best-effort "doc marker" from existing chunks.

        This is used for incremental indexing to detect unchanged documents.

        Args:
            doc_id: Document id.

        Returns:
            A dict with keys like ``doc_hash`` and ``source_md_mtime_ns`` if present,
            or None if the document is not found in the collection.
        """
        doc_id_s = str(doc_id)
        got = self._collection.get(where={"doc_id": doc_id_s}, limit=1, include=["metadatas"])
        metas = got.get("metadatas") or []
        if not metas:
            return None
        meta0 = metas[0] or {}
        if not isinstance(meta0, dict):
            return None
        return meta0

    def list_unique_doc_ids(self, *, page_size: int = 2000) -> Set[str]:
        """List unique doc_ids by scanning metadatas (O(n))."""
        total = int(self.collection_count())
        if total <= 0:
            return set()
        out: Set[str] = set()
        offset = 0
        page = max(1, int(page_size))
        while offset < total:
            metas = self.get_metadatas_page(offset=offset, limit=page)
            if not metas:
                break
            for meta in metas:
                did = meta.get("doc_id")
                if did is not None and str(did):
                    out.add(str(did))
            offset += len(metas)
            if len(metas) < page:
                break
        return out

    def reset_collection(self) -> None:
        """Drop and recreate current collection (rebuild indexing)."""
        name = self.collection_name
        try:
            self._client.delete_collection(name=name)
        except Exception:
            # If collection doesn't exist or backend differs, ignore.
            pass
        self._collection = self._client.get_or_create_collection(name=name)

    def upsert_chunks(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, object]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self._collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
            embeddings=list(embeddings),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: Optional[Dict[str, object]] = None,
    ) -> List[SearchHit]:
        """Search by query embedding with optional metadata filter.

        Args:
            query: Query text.
            top_k: Number of results.
            where: Optional Chroma where filter, e.g. {"doc_id": "paper1"}.

        Returns:
            List of search hits.
        """
        if self._embedder is None:
            raise RuntimeError("Embedder is not configured for this ChromaStore instance.")
        # Avoid importing BgeM3Embedder here to keep store usable without embeddings.
        q_emb = getattr(self._embedder, "embed_query")(query)
        res = self._collection.query(
            query_embeddings=[q_emb],
            n_results=int(top_k),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        hits: List[SearchHit] = []
        for rid, doc_text, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            # Prefer Chroma-returned id; fall back to metadata if needed.
            chunk_id = str(rid or meta.get("chunk_id") or "")
            doc_id = str(meta.get("doc_id") or "")
            short_name = str(meta.get("short_name") or doc_id)
            section = meta.get("section")
            section_s = str(section) if section is not None else None
            source_md = str(meta.get("source_md") or "")
            preview = (doc_text or "")[:400]
            # Chroma distance: smaller is more similar for cosine/.. depending on settings.
            # For MVP, expose score as (1 - distance) when distance looks like cosine distance.
            try:
                score = float(1.0 - float(dist))
            except Exception:
                score = float("nan")
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    score=score,
                    doc_id=doc_id,
                    short_name=short_name,
                    section=section_s,
                    preview=preview,
                    source_md=source_md,
                )
            )
        return hits

    def search_by_embedding(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: Optional[Dict[str, object]] = None,
    ) -> List[SearchHit]:
        """Search by a precomputed query embedding.

        This is useful when the caller wants to reuse the same query vector across
        multiple stores/collections without recomputing embeddings.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.
            where: Optional Chroma where filter.

        Returns:
            List of search hits.
        """

        q_emb = list(query_embedding)
        res = self._collection.query(
            query_embeddings=[q_emb],
            n_results=int(top_k),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        hits: List[SearchHit] = []
        for rid, doc_text, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            chunk_id = str(rid or meta.get("chunk_id") or "")
            doc_id = str(meta.get("doc_id") or "")
            short_name = str(meta.get("short_name") or doc_id)
            section = meta.get("section")
            section_s = str(section) if section is not None else None
            source_md = str(meta.get("source_md") or "")
            preview = (doc_text or "")[:400]
            try:
                score = float(1.0 - float(dist))
            except Exception:
                score = float("nan")
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    score=score,
                    doc_id=doc_id,
                    short_name=short_name,
                    section=section_s,
                    preview=preview,
                    source_md=source_md,
                )
            )
        return hits

    def _get_all_chunks_for_doc(self, doc_id: str) -> List[ChunkContext]:
        got = self._collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []

        items: List[ChunkContext] = []
        for _id, text, meta in zip(ids, docs, metas):
            meta = meta or {}
            items.append(
                ChunkContext(
                    chunk_id=str(meta.get("chunk_id") or _id),
                    doc_id=str(meta.get("doc_id") or doc_id),
                    short_name=str(meta.get("short_name") or meta.get("doc_id") or doc_id),
                    source_md=str(meta.get("source_md") or ""),
                    section=str(meta.get("section")) if meta.get("section") is not None else None,
                    text=str(text or ""),
                    chunk_index=int(meta.get("chunk_index") or 0),
                )
            )
        items.sort(key=lambda x: x.chunk_index)
        return items

    def get_context_by_chunk_id(self, chunk_id: str, *, window: int = 1) -> Dict[str, Any]:
        got = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not got.get("ids"):
            raise KeyError(f"chunk_id not found: {chunk_id}")
        meta = (got.get("metadatas") or [None])[0] or {}
        doc_id = str(meta.get("doc_id") or "")
        short_name = str(meta.get("short_name") or doc_id)
        source_md = str(meta.get("source_md") or "")
        section = str(meta.get("section")) if meta.get("section") is not None else None
        chunk_index = int(meta.get("chunk_index") or 0)

        all_chunks = self._get_all_chunks_for_doc(doc_id)
        if not all_chunks:
            text = str((got.get("documents") or [""])[0] or "")
            return {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "short_name": short_name,
                "source_md": source_md,
                "section": section,
                "chunk_index": chunk_index,
                "text": text,
            }

        start = max(0, chunk_index - int(window))
        end = min(len(all_chunks), chunk_index + int(window) + 1)
        merged = "\n\n---\n\n".join(c.text for c in all_chunks[start:end] if c.text)

        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "short_name": short_name,
            "source_md": source_md,
            "section": section,
            "chunk_index": chunk_index,
            "window": int(window),
            "text": merged,
        }
