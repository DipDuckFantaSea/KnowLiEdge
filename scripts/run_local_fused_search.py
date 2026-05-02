from __future__ import annotations

"""Run the same local fused search as MCP ``search_knowledge_base_fused`` / ``universal_academic_search``.

Used for CLI / automation when MCP is not attached. Mirrors ``create_mcp_app`` logic in
``knotliedge.mcp.server`` for the fused local branch.
"""

import argparse
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from knotliedge.config.load import load_app_config
from knotliedge.embeddings.lazy_ipc_gate import LazyIpcEmbedderSession
from knotliedge.retrieval.rrf import RankedId, rrf_merge
from knotliedge.storage.chroma_store import ChromaStore
from knotliedge.storage.fts_store import FtsStore, default_fts_db_path


def _extract_keyphrase(q: str) -> str:
    s = (q or "").strip()
    if not s:
        return ""
    m = re.search(r"\"([^\"]{2,200})\"", s)
    if m:
        return m.group(1).strip()
    spans = re.findall(r"[A-Za-z0-9][A-Za-z0-9 +/\\\\-]{2,120}", s)
    if spans:
        spans.sort(key=len, reverse=True)
        return spans[0].strip()
    return s


def run_fused_local_search(
    *,
    cfg,
    config_path: Path,
    query: str,
    top_k: int,
    top_k_fts: int,
    top_k_vec: int,
    rrf_k: int,
    doc_id: Optional[str],
    short_name: Optional[str],
    fts_mode: str,
) -> Dict[str, Any]:
    embed_gate = LazyIpcEmbedderSession(config_path=config_path)
    store = ChromaStore(cfg=cfg, embedder=None)

    q = (query or "").strip()
    if not q:
        return {"query": query, "top_k": int(top_k), "results": []}

    where: Optional[Dict[str, object]] = None
    if doc_id:
        where = {"doc_id": str(doc_id)}
    elif short_name:
        where = {"short_name": str(short_name)}

    keyphrase = _extract_keyphrase(q)

    def _embed_query_once(text: str) -> Sequence[float]:
        last_err: Optional[BaseException] = None
        for attempt in range(2):
            try:
                embed_gate.require_for_store(store)
                emb = embed_gate.get_optional()
                if emb is None:
                    raise RuntimeError("Embedder is not available after require_for_store().")
                return getattr(emb, "embed_query")(text)
            except Exception as e:
                last_err = e
                if attempt == 0:
                    try:
                        embed_gate.invalidate()
                    except Exception:
                        pass
                else:
                    break
        raise RuntimeError(f"embed_query failed after retry: {last_err}") from last_err

    q_emb = _embed_query_once(q)

    def run_fts() -> List[Any]:
        fts_store = FtsStore(db_path=default_fts_db_path(cfg))
        mode = (fts_mode or "bm25").strip().lower()
        if mode == "literal":
            return fts_store.search_literal_substring(
                keyphrase, top_k=int(top_k_fts), where=where, case_sensitive=True
            )
        return fts_store.search(q, top_k=int(top_k_fts), where=where)

    def run_vec() -> List[Any]:
        return store.search_by_embedding(q_emb, top_k=int(top_k_vec), where=where)

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        f_fts = ex.submit(run_fts)
        f_vec = ex.submit(run_vec)
        fts_hits = f_fts.result()
        vec_hits = f_vec.result()

    fts_ranked = [RankedId(chunk_id=h.chunk_id, rank=i + 1, score=h.score) for i, h in enumerate(fts_hits)]
    vec_ranked = [RankedId(chunk_id=h.chunk_id, rank=i + 1, score=h.score) for i, h in enumerate(vec_hits)]

    merged = rrf_merge(fts=fts_ranked, vec=vec_ranked, rrf_k=int(rrf_k), limit=int(top_k))

    vec_by_id = {h.chunk_id: h for h in vec_hits}
    fts_by_id = {h.chunk_id: h for h in fts_hits}
    meta_map = FtsStore(db_path=default_fts_db_path(cfg)).get_chunk_meta_map([m.chunk_id for m in merged])

    out_results: List[Dict[str, Any]] = []
    for m in merged:
        cid = m.chunk_id
        vh = vec_by_id.get(cid)
        fh = fts_by_id.get(cid)
        meta = meta_map.get(cid, {})
        out_results.append(
            {
                "chunk_id": cid,
                "rrf_score": float(m.rrf_score),
                "fts_rank": m.fts_rank,
                "vec_rank": m.vec_rank,
                "fts_score": m.fts_score,
                "vec_score": m.vec_score,
                "doc_id": (vh.doc_id if vh else (fh.doc_id if fh else "")),
                "short_name": (vh.short_name if vh else (fh.short_name if fh else "")),
                "section": (vh.section if vh else (fh.section if fh else None)),
                "preview": (fh.preview if fh else (vh.preview if vh else "")),
                "source_md": (vh.source_md if vh else str(meta.get("source_md") or "")),
                "chunk_index": (int(meta.get("chunk_index") or 0) if meta else None),
            }
        )

    return {
        "query": query,
        "top_k": int(top_k),
        "rrf_k": int(rrf_k),
        "filter": {"doc_id": doc_id, "short_name": short_name},
        "fts_enabled": True,
        "vector_enabled": True,
        "fts_mode": (fts_mode or "bm25").strip().lower(),
        "keyphrase": keyphrase,
        "results": out_results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Local fused FTS+vector RRF search (MCP-equivalent).")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--top-k-fts", type=int, default=40)
    p.add_argument("--top-k-vec", type=int, default=40)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--doc-id", type=str, default=None)
    p.add_argument("--short-name", type=str, default=None)
    p.add_argument("--fts-mode", type=str, default="bm25", choices=["bm25", "literal"])
    p.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON results.")
    args = p.parse_args()

    config_path = Path(args.config)
    cfg = load_app_config(config_path)
    payload = run_fused_local_search(
        cfg=cfg,
        config_path=config_path,
        query=args.query,
        top_k=int(args.top_k),
        top_k_fts=int(args.top_k_fts),
        top_k_vec=int(args.top_k_vec),
        rrf_k=int(args.rrf_k),
        doc_id=args.doc_id,
        short_name=args.short_name,
        fts_mode=args.fts_mode,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
