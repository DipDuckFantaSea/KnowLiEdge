"""ChromaDB collection summary and search smoke helpers shared by MCP and CLI."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set

from knotliedge.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

_DEFAULT_PAGE_SIZE = 2000


def summarize_chroma_collection(
    store: ChromaStore,
    *,
    page_size: int = _DEFAULT_PAGE_SIZE,
) -> Dict[str, Any]:
    """Aggregate collection statistics from stored chunk metadatas.

    Scans all chunks in pages (O(n) in collection size) to compute
    ``latest_created_at`` and exact ``unique_doc_ids``.

    Args:
        store: Open ChromaStore with a live collection.
        page_size: Number of rows to fetch per Chroma ``get`` call.

    Returns:
        Dict with keys: ``collection_name``, ``collection_count``,
        ``latest_created_at`` (ISO8601 string or ``None`` if empty / no timestamps),
        ``unique_doc_ids`` (int).
    """
    name = store.collection_name
    try:
        total = int(store.collection_count())
    except Exception as e:
        logger.error("collection_count failed for %s: %s", name, e)
        return {
            "collection_name": name,
            "collection_count": -1,
            "latest_created_at": None,
            "unique_doc_ids": 0,
            "scan_error": str(e),
        }

    if total <= 0:
        return {
            "collection_name": name,
            "collection_count": total,
            "latest_created_at": None,
            "unique_doc_ids": 0,
        }

    latest: Optional[str] = None
    doc_ids: Set[str] = set()
    offset = 0
    page = max(1, int(page_size))

    while offset < total:
        try:
            metas = store.get_metadatas_page(offset=offset, limit=page)
        except Exception as e:
            logger.error("get_metadatas_page failed offset=%s limit=%s: %s", offset, page, e)
            return {
                "collection_name": name,
                "collection_count": total,
                "latest_created_at": latest,
                "unique_doc_ids": len(doc_ids),
                "scan_error": str(e),
            }
        if not metas:
            break
        for meta in metas:
            did = meta.get("doc_id")
            if did is not None and str(did):
                doc_ids.add(str(did))
            ca = meta.get("created_at")
            if ca is not None and str(ca):
                s = str(ca)
                if latest is None or s > latest:
                    latest = s
        offset += len(metas)
        if len(metas) < page:
            break

    return {
        "collection_name": name,
        "collection_count": total,
        "latest_created_at": latest,
        "unique_doc_ids": len(doc_ids),
    }


def smoke_search_hits(store: ChromaStore, query: str, top_k: int) -> List[Dict[str, Any]]:
    """Run a small vector search and return JSON-friendly hit summaries.

    Args:
        store: ChromaStore with embedder-backed search.
        query: Query text.
        top_k: Number of hits.

    Returns:
        List of dicts with chunk_id, score, doc_id, short_name, section, preview.
    """
    hits = store.search(str(query), top_k=int(top_k))
    out: List[Dict[str, Any]] = []
    for h in hits:
        try:
            sv = float(h.score)
            score_out: Optional[float] = None if math.isnan(sv) else sv
        except (TypeError, ValueError):
            score_out = None
        out.append(
            {
                "chunk_id": h.chunk_id,
                "score": score_out,
                "doc_id": h.doc_id,
                "short_name": h.short_name,
                "section": h.section,
                "preview": (h.preview or "").replace("\n", " ")[:200],
            }
        )
    return out
