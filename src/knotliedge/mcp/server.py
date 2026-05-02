from __future__ import annotations

import concurrent.futures as cf
import base64
import mimetypes
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fastmcp import FastMCP

from knotliedge.config.load import load_app_config
from knotliedge.embeddings.lazy_ipc_gate import LazyIpcEmbedderSession
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.mineru.http_client import get_task_result, get_task_status, submit_task
from knotliedge.mineru.service import start_service, status as mineru_status, stop_service
from knotliedge.ops.runtime_paths import get_runtime_paths
from knotliedge.ops.chroma_health import smoke_search_hits, summarize_chroma_collection
from knotliedge.citation_graph.manager import CitationGraphManager
from knotliedge.citation_graph.store import CitationGraphStore, default_citation_db_path
from knotliedge.retrieval.rrf import RankedId, rrf_merge
from knotliedge.mcp.hybrid_knowledge_radar_search import run_hybrid_knowledge_and_radar_search
from knotliedge.storage.chroma_store import ChromaStore
from knotliedge.storage.fts_store import FtsStore, default_fts_db_path
from knotliedge.storage.venue_radar_store import VenueRadarStore, default_venue_radar_db_path
from knotliedge.venue_radar.radar import run_venue_radar
from knotliedge.workflow.planning import build_and_write_workflow_plan_markdown
from knotliedge.executors.base import EvidenceInput
from knotliedge.executors.summarize_with_citations import run_summarize_with_citations
from knotliedge.executors.compare_papers_by_fields import run_compare_papers_by_fields

logger = setup_logging()


def create_mcp_app(*, config_path: Path) -> FastMCP:
    """Create MCP app with two-step knowledge base tools.

    Args:
        config_path: Path to YAML config.

    Returns:
        FastMCP app.
    """
    cfg = load_app_config(config_path)
    # Defer get_embedder (and GPU embedding server autostart) until a tool needs vectors.
    embed_gate = LazyIpcEmbedderSession(config_path=config_path)
    store = ChromaStore(cfg=cfg, embedder=None)

    mcp = FastMCP("KnotLiEdge")

    def _extract_keyphrase(q: str) -> str:
        """Extract a best-effort literal keyphrase from a natural-language query.

        Priority:
        - First quoted segment: "GaN on SiN"
        - Else longest ASCII-ish technical span: letters/digits with separators
        - Else fallback to the original query
        """

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

    def _embed_query_once(q: str) -> Sequence[float]:
        last_err: Optional[BaseException] = None
        for attempt in range(2):
            try:
                embed_gate.require_for_store(store)
                emb = embed_gate.get_optional()
                if emb is None:
                    raise RuntimeError("Embedder is not available after require_for_store().")
                return getattr(emb, "embed_query")(q)
            except Exception as e:
                last_err = e
                if attempt == 0:
                    logger.warning("embed_query failed; invalidate session and retry once | %s", e)
                    try:
                        embed_gate.invalidate()
                    except Exception:
                        pass
                else:
                    break
        raise RuntimeError(f"embed_query failed after retry: {last_err}") from last_err

    @mcp.tool()
    def ping() -> Dict[str, Any]:
        """Lightweight health check (no Chroma scan).

        Returns:
            Basic service info and config identifiers.
        """
        return {
            "name": "KnotLiEdge",
            "ready": True,
            "embedding_ready": embed_gate.ready,
            "config_path": str(config_path),
            "collection_name": str(cfg.chroma.collection_name),
            "chroma_http_host": str(cfg.chroma.http_host),
            "chroma_http_port": int(cfg.chroma.http_port),
            "chroma_persist_dir": str(cfg.paths.chroma_db_dir),
        }

    @mcp.tool()
    def stats(
        smoke_query: Optional[str] = None,
        smoke_top_k: int = 3,
    ) -> Dict[str, Any]:
        """Return knowledge base statistics; optional vector search smoke test.

        When ``smoke_query`` is non-empty, runs one ``search`` and includes hits
        under ``smoke`` (embedder required). Default omits smoke to avoid extra
        embedding work on every call.

        Args:
            smoke_query: If set, run a smoke ``search`` with this text.
            smoke_top_k: Top-k for smoke search when ``smoke_query`` is set.

        Returns:
            Stats dict including paths, collection summary, and optional ``smoke``.
        """
        base_light = {
            "config_path": str(config_path),
            "embedding_ready": embed_gate.ready,
            "project_root": str(cfg.project_root),
            "chroma_db_dir": str(cfg.paths.chroma_db_dir),
            "chroma_http": f"{cfg.chroma.http_host}:{cfg.chroma.http_port}",
            "collection_name": str(cfg.chroma.collection_name),
            "markdown_vault_dir": str(cfg.paths.markdown_vault_dir),
            "embedding_model_name_or_path": str(cfg.embedding.model_name_or_path),
            "embedding_device": str(cfg.embedding.device),
        }
        # store is always available; only vector search requires embedder readiness.
        try:
            count = int(store.collection_count())
        except Exception as e:
            count = -1
            logger.error("Failed to get collection count: %s", e)
        out: Dict[str, Any] = {
            **base_light,
            "ready": True,
            "collection_count": count,
        }
        if count < 0:
            return out

        summary = summarize_chroma_collection(store)
        out["collection_name"] = summary.get("collection_name", out["collection_name"])
        out["collection_count"] = summary.get("collection_count", count)
        out["latest_created_at"] = summary.get("latest_created_at")
        out["unique_doc_ids"] = summary.get("unique_doc_ids")
        if summary.get("scan_error"):
            out["scan_error"] = summary["scan_error"]

        q = (smoke_query or "").strip()
        if q:
            try:
                embed_gate.require_for_store(store)
                out["smoke"] = {
                    "query": q,
                    "top_k": int(smoke_top_k),
                    "hits": smoke_search_hits(store, q, int(smoke_top_k)),
                }
            except Exception as e:
                logger.error("stats smoke search failed query=%s: %s", q, e)
                out["smoke"] = {"query": q, "error": str(e)}
        return out

    @mcp.tool()
    def search_knowledge_base(
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        short_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search the local knowledge base and return candidate chunks.

        Args:
            query: User query.
            top_k: Number of results.
            doc_id: Optional filter to restrict search within a document id.
            short_name: Optional filter to restrict search within a short name.

        Returns:
            Dict with results list.
        """
        embed_gate.require_for_store(store)
        where: Optional[Dict[str, object]] = None
        if doc_id:
            where = {"doc_id": str(doc_id)}
        elif short_name:
            where = {"short_name": str(short_name)}

        hits = store.search(query, top_k=top_k, where=where)
        return {
            "query": query,
            "top_k": int(top_k),
            "filter": {"doc_id": doc_id, "short_name": short_name},
            "results": [asdict(h) for h in hits],
        }

    @mcp.tool()
    def search_knowledge_base_fused(
        query: str,
        top_k: int = 5,
        top_k_fts: int = 20,
        top_k_vec: int = 20,
        rrf_k: int = 60,
        doc_id: Optional[str] = None,
        short_name: Optional[str] = None,
        fts_mode: str = "bm25",
    ) -> Dict[str, Any]:
        """Fused search: FTS5 keyword + vector search merged by RRF.

        Args:
            query: User query.
            top_k: Final merged top-k.
            top_k_fts: Keyword search candidates.
            top_k_vec: Vector search candidates.
            rrf_k: RRF constant (larger reduces head dominance).
            doc_id: Optional filter to restrict within doc_id.
            short_name: Optional filter to restrict within short_name.
            fts_mode: One of {"bm25","literal"}. Default "bm25" for efficiency.

        Returns:
            Dict with merged results and per-source ranks/scores.
        """
        q = (query or "").strip()
        if not q:
            return {"query": query, "top_k": int(top_k), "results": []}

        where: Optional[Dict[str, object]] = None
        if doc_id:
            where = {"doc_id": str(doc_id)}
        elif short_name:
            where = {"short_name": str(short_name)}

        keyphrase = _extract_keyphrase(q)
        q_emb = _embed_query_once(q)

        def run_fts() -> List[Any]:
            fts_store = FtsStore(db_path=default_fts_db_path(cfg))
            mode = (fts_mode or "bm25").strip().lower()
            if mode == "literal":
                # Optional strict literal substring search + highlight.
                return fts_store.search_literal_substring(
                    keyphrase, top_k=int(top_k_fts), where=where, case_sensitive=True
                )
            # Default: BM25 keyword search (fast, good recall).
            return fts_store.search(q, top_k=int(top_k_fts), where=where)

        def run_vec() -> List[Any]:
            # Vector branch: use precomputed embedding.
            return store.search_by_embedding(q_emb, top_k=int(top_k_vec), where=where)

        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            f_fts = ex.submit(run_fts)
            f_vec = ex.submit(run_vec)
            fts_hits = f_fts.result()
            vec_hits = f_vec.result()

        fts_ranked = [RankedId(chunk_id=h.chunk_id, rank=i + 1, score=h.score) for i, h in enumerate(fts_hits)]
        vec_ranked = [RankedId(chunk_id=h.chunk_id, rank=i + 1, score=h.score) for i, h in enumerate(vec_hits)]

        merged = rrf_merge(fts=fts_ranked, vec=vec_ranked, rrf_k=int(rrf_k), limit=int(top_k))

        # Hydrate fields
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
                    # Prefer FTS preview; when fts_mode=literal it contains highlight markers.
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

    @mcp.tool()
    def search_hybrid_knowledge_and_radar(
        query: str,
        *,
        top_k_total: int = 20,
        top_k_local_docs: int = 10,
        top_k_radar_works: int = 10,
        local_chunks_per_doc: int = 3,
        top_k_fts: int = 80,
        top_k_vec: int = 80,
        rrf_k: int = 60,
        ai_parse_query: bool = True,
    ) -> Dict[str, Any]:
        """Dual-wield search: local deep KB (FTS5+Chroma RRF) + venue radar (vector-only).

        Args:
            query: User query.
            top_k_local: Top-k to return from local deep knowledge base.
            top_k_radar: Top-k to return from venue radar quarantine store.
            top_k_fts: Candidate size for local FTS5.
            top_k_vec: Candidate size for local vector search.
            rrf_k: RRF constant for local fusion.
            ai_parse_query: If True (default), use OpenAI-compatible Chat Completions
                (hybrid path loads ``templates/openai_chat/local_research_keyword_en_terms.json``;
                model ``deepseek-v4-flash`` via DeepSeek endpoint)
                to expand ``query`` into space-separated keywords before local multi-keyword fusion;
                requires ``DEEPSEEK_API_KEY`` or ``OPENAI_API_KEY`` in root ``.env``. Set to ``False`` to use heuristic splitting only. On LLM failure, falls back to heuristic splitting.

        Returns:
            Dict including a markdown string under `markdown`.
        """

        return run_hybrid_knowledge_and_radar_search(
            cfg,
            store,
            _embed_query_once,
            query,
            top_k_total=top_k_total,
            top_k_local_docs=top_k_local_docs,
            top_k_radar_works=top_k_radar_works,
            local_chunks_per_doc=local_chunks_per_doc,
            top_k_fts=top_k_fts,
            top_k_vec=top_k_vec,
            rrf_k=rrf_k,
            report_markdown_dir=default_venue_radar_db_path(cfg).parent,
            ai_parse_query=ai_parse_query,
        )

    @mcp.tool()
    def venue_radar_stats(top_n: int = 20) -> Dict[str, Any]:
        """Return venue radar quarantine store stats (SQLite).

        Args:
            top_n: Max rows to return for duplicate lists.

        Returns:
            Stats dict.
        """

        radar_db = VenueRadarStore(db_path=default_venue_radar_db_path(cfg))
        out = radar_db.get_stats(top_n=int(top_n))
        out["config_path"] = str(config_path)
        out["ready"] = True
        return out

    @mcp.tool()
    def venue_radar_sync(
        *,
        lookback_days: int = 30,
        limit: Optional[int] = None,
        purge: bool = False,
        mark_local: bool = True,
    ) -> Dict[str, Any]:
        """Sync recent OpenAlex works for configured venues into radar stores.

        This is an *incremental upsert* keyed by OpenAlex work id, so re-running within
        the same lookback window should not create duplicates.

        Args:
            lookback_days: Only fetch works published within past N days.
            limit: Optional max works to ingest (across venues).
            purge: If True, purge radar stores before fetching.
            mark_local: If True, scan local vault and mark presence flags.

        Returns:
            Dict with ingest count and post-sync stats snapshot.
        """

        before = venue_radar_stats(top_n=10)
        ingested = int(
            run_venue_radar(
                config_path=Path(config_path),
                limit=int(limit) if limit is not None else None,
                no_fetch=False,
                purge=bool(purge),
                mark_local=bool(mark_local),
                lookback_days=int(lookback_days),
            )
        )
        after = venue_radar_stats(top_n=10)
        return {
            "ready": True,
            "config_path": str(config_path),
            "lookback_days": int(lookback_days),
            "limit": (int(limit) if limit is not None else None),
            "purge": bool(purge),
            "mark_local": bool(mark_local),
            "ingested": ingested,
            "before": before,
            "after": after,
        }

    @mcp.tool()
    def universal_academic_search(
        query: str,
        *,
        mode: str = "auto",
        fused: bool = True,
        use_radar: bool = False,
        top_k: int = 10,
        top_k_total: int = 20,
        top_k_local_docs: int = 10,
        top_k_radar_works: int = 10,
        ai_parse_query: bool = True,
    ) -> Dict[str, Any]:
        """Unified search entrypoint to reduce tool-selection cognitive load.

        This tool intentionally wraps existing tools instead of replacing them.

        Args:
            query: User query.
            mode: One of {"auto","local","hybrid"}. ``auto`` chooses based on flags.
            fused: If True, prefer fused local search (FTS+vec RRF) for local mode.
            use_radar: When True, choose hybrid mode in ``auto``.
            top_k: Local-only output top_k (used for fused local search).
            top_k_total: Hybrid total items (used for hybrid report).
            top_k_local_docs: Hybrid local docs.
            top_k_radar_works: Hybrid radar works.
            ai_parse_query: Pass-through for hybrid keyword expansion.

        Returns:
            A unified dict with keys ``kind`` and either ``local`` or ``hybrid`` payload.
        """

        m = (mode or "auto").strip().lower()
        if m not in {"auto", "local", "hybrid"}:
            raise ValueError('mode must be one of: "auto", "local", "hybrid"')

        chosen = m
        if chosen == "auto":
            chosen = "hybrid" if bool(use_radar) else "local"

        if chosen == "hybrid":
            payload = search_hybrid_knowledge_and_radar(
                query,
                top_k_total=int(top_k_total),
                top_k_local_docs=int(top_k_local_docs),
                top_k_radar_works=int(top_k_radar_works),
                ai_parse_query=bool(ai_parse_query),
            )
            return {"kind": "hybrid", "query": query, "hybrid": payload}

        if bool(fused):
            payload = search_knowledge_base_fused(query=query, top_k=int(top_k))
            return {"kind": "local_fused", "query": query, "local": payload}

        payload = search_knowledge_base(query=query, top_k=int(top_k))
        return {"kind": "local_vec", "query": query, "local": payload}

    @mcp.tool()
    def get_knowledge_chunk(chunk_id: str, window: int = 1) -> Dict[str, Any]:
        """Get expanded context for a chunk id.

        Args:
            chunk_id: Chunk id from search results.
            window: Number of neighbor chunks to include on each side.

        Returns:
            Context dict with `text` and file locator.
        """
        pack = store.get_context_by_chunk_id(chunk_id, window=window)
        did = str(pack.get("doc_id") or "").strip()
        prof = ""
        if did:
            try:
                prof = FtsStore(db_path=default_fts_db_path(cfg)).get_document_profile(doc_id=did)
            except Exception:
                prof = ""
        pack["document_profile"] = prof
        pack["prompt_text"] = "\n".join(
            [
                "[全局文献特征]",
                prof or "",
                "",
                "[相关局部细节]",
                str(pack.get("text") or ""),
            ]
        ).strip() + "\n"
        return pack

    def _path_is_under(child: Path, parent: Path) -> bool:
        try:
            return child.is_relative_to(parent)  # type: ignore[attr-defined]
        except Exception:
            try:
                child.resolve().relative_to(parent.resolve())
                return True
            except Exception:
                return False

    @mcp.tool()
    def read_local_image_as_base64(
        path: str,
        *,
        max_bytes: int = 5_000_000,
    ) -> Dict[str, Any]:
        """Read a local vault image and return base64 + mime for multimodal LLMs.

        Security boundary: only reads under configured markdown vault roots.
        """

        p = Path(str(path or "")).expanduser()
        if not p.is_absolute():
            # Interpret relative paths as relative to markdown_vault_dir (most common).
            p = (Path(cfg.paths.markdown_vault_dir) / p).resolve()
        else:
            p = p.resolve()

        vault_root = Path(cfg.paths.markdown_vault_dir).resolve()
        assets_root = Path(cfg.paths.markdown_assets_dir).resolve()
        if not (_path_is_under(p, vault_root) or _path_is_under(p, assets_root)):
            raise ValueError(f"Refuse to read image outside vault/assets roots: {p}")
        if not p.is_file():
            raise FileNotFoundError(str(p))

        size = int(p.stat().st_size)
        if size <= 0:
            raise ValueError(f"Empty image file: {p}")
        if size > int(max_bytes):
            raise ValueError(f"Image too large ({size} bytes > max_bytes={int(max_bytes)}): {p}")

        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        mime, _enc = mimetypes.guess_type(str(p))
        if not mime:
            # best-effort by suffix
            suf = p.suffix.lower()
            if suf in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suf == ".png":
                mime = "image/png"
            elif suf == ".gif":
                mime = "image/gif"
            elif suf == ".webp":
                mime = "image/webp"
            elif suf == ".svg":
                mime = "image/svg+xml"
            else:
                mime = "application/octet-stream"

        width: Optional[int] = None
        height: Optional[int] = None
        try:
            from PIL import Image  # type: ignore[import-not-found]

            with Image.open(p) as im:
                width, height = int(im.size[0]), int(im.size[1])
        except Exception:
            width, height = None, None

        return {
            "path": str(p),
            "bytes": size,
            "mime": mime,
            "b64": b64,
            "width": width,
            "height": height,
        }

    @mcp.tool()
    def list_images_near_chunk(chunk_id: str, *, max_n: int = 12) -> Dict[str, Any]:
        """List candidate images under ``assets/<short_name>/images`` for a chunk."""

        pack = store.get_context_by_chunk_id(str(chunk_id), window=0)
        short_name = str(pack.get("short_name") or "").strip()
        source_md = str(pack.get("source_md") or "").strip()
        if not short_name:
            return {"chunk_id": chunk_id, "short_name": short_name, "source_md": source_md, "images": []}

        assets_root = Path(cfg.paths.markdown_assets_dir).resolve()
        img_dir = (assets_root / short_name / "images").resolve()
        if not img_dir.exists() or not img_dir.is_dir():
            return {"chunk_id": chunk_id, "short_name": short_name, "source_md": source_md, "images": []}

        exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
        items = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        items.sort(key=lambda x: x.name)
        out: List[Dict[str, Any]] = []
        for p in items[: max(0, int(max_n))]:
            out.append({"path": str(p), "name": p.name, "bytes": int(p.stat().st_size)})
        return {"chunk_id": chunk_id, "short_name": short_name, "source_md": source_md, "images": out}

    @mcp.tool()
    def summarize_with_citations(
        chunk_ids: List[str],
        *,
        window: int = 2,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evidence-grounded extractive summary with strict citations (M2 executor, MVP)."""

        ev = EvidenceInput(chunk_ids=list(chunk_ids or []), evidence_refs=[])
        res = run_summarize_with_citations(
            project_root=cfg.project_root,
            evidence=ev,
            get_chunk_context=lambda cid, w: store.get_context_by_chunk_id(cid, window=w),
            window=int(window),
            run_id=run_id,
        )
        return asdict(res)

    @mcp.tool()
    def compare_papers_by_fields(
        papers: List[Dict[str, Any]],
        *,
        fields: Optional[List[str]] = None,
        window: int = 1,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evidence-grounded paper comparison table with strict citations (M2 executor, MVP)."""

        res = run_compare_papers_by_fields(
            project_root=cfg.project_root,
            papers=list(papers or []),
            fields=list(fields or []) if fields is not None else None,
            get_chunk_context=lambda cid, w: store.get_context_by_chunk_id(cid, window=w),
            window=int(window),
            run_id=run_id,
        )
        return asdict(res)

    @mcp.tool()
    def mineru_service_start(
        enable_vlm_preload: bool = False,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start a persistent mineru-api service.

        VLM preload is **off** by default: preloading can block startup for a long time
        (lmdeploy init) so ``/health`` never becomes available and clients see connection
        refused. Set ``enable_vlm_preload=True`` only if you need VLM paths and can wait.

        Environment matches ``watch_ingest`` local service: ``MINERU_MODEL_SOURCE=local`` and
        ``MINERU_VIRTUAL_VRAM_SIZE=8`` (GB-scale for this MinerU build).
        """
        rp = get_runtime_paths(cfg)
        st = start_service(
            enable_vlm_preload=bool(enable_vlm_preload),
            host=str(host),
            port=int(port) if port else None,
            env_overrides={
                "MINERU_MODEL_SOURCE": "local",
                "MINERU_VIRTUAL_VRAM_SIZE": "8",
            },
            log_dir=rp.mineru_logs_dir,
            work_dir=rp.mineru_work_dir,
        )
        return {
            "api_url": st.api_url,
            "pid": int(st.pid),
            "enable_vlm_preload": bool(st.enable_vlm_preload),
            "host": st.host,
            "port": int(st.port),
        }

    @mcp.tool()
    def mineru_service_status() -> Dict[str, Any]:
        """Return current mineru-api status (pid/api_url)."""
        return mineru_status()

    @mcp.tool()
    def mineru_service_stop() -> Dict[str, Any]:
        """Stop mineru-api service if running."""
        return stop_service()

    @mcp.tool()
    def mineru_submit_parse_task(
        pdf_path: str,
        backend: str = "hybrid-auto-engine",
        parse_method: str = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        return_md: bool = True,
        api_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit an async parse task to mineru-api and return task_id."""
        st = mineru_status()
        use_api_url = str(api_url or st.get("api_url") or "")
        if not use_api_url:
            raise RuntimeError("mineru-api not running. Call mineru_service_start first or pass api_url.")
        submitted = submit_task(
            api_url=use_api_url,
            pdf_paths=[Path(pdf_path)],
            backend=str(backend),
            parse_method=str(parse_method),
            formula_enable=bool(formula_enable),
            table_enable=bool(table_enable),
            return_md=bool(return_md),
        )
        return {"api_url": use_api_url, "task_id": submitted.task_id}

    @mcp.tool()
    def mineru_get_task_status(task_id: str, api_url: Optional[str] = None) -> Dict[str, Any]:
        st = mineru_status()
        use_api_url = str(api_url or st.get("api_url") or "")
        if not use_api_url:
            raise RuntimeError("mineru-api not running. Call mineru_service_start first or pass api_url.")
        return {"api_url": use_api_url, "task_id": task_id, "status": get_task_status(api_url=use_api_url, task_id=task_id)}

    @mcp.tool()
    def mineru_get_task_result(task_id: str, api_url: Optional[str] = None) -> Dict[str, Any]:
        st = mineru_status()
        use_api_url = str(api_url or st.get("api_url") or "")
        if not use_api_url:
            raise RuntimeError("mineru-api not running. Call mineru_service_start first or pass api_url.")
        return {"api_url": use_api_url, "task_id": task_id, "result": get_task_result(api_url=use_api_url, task_id=task_id)}

    @mcp.tool()
    def get_document_lineage(doc_id: str) -> str:
        """Summarize OpenAlex one-hop citation lineage for a vault document (Markdown for LLM).

        Args:
            doc_id: Local knowledge base ``doc_id`` (must align with FTS ``documents``).

        Returns:
            Markdown text listing referenced works and citing works with titles and ``short_name`` hints.
        """
        mgr = CitationGraphManager(cfg=cfg)
        return mgr.get_lineage(str(doc_id))

    @mcp.tool()
    def trace_research_evolution(start_id: str, end_id: str) -> str:
        """Trace a shortest conceptual chain between two works in the undirected citation graph.

        Args:
            start_id: OpenAlex work URL/id, or ``doc:{doc_id}`` when linked via FTS ``openalex_id``.
            end_id: Same format as ``start_id``.

        Returns:
            Markdown chain (titles / short labels); not a JSON payload.
        """
        mgr = CitationGraphManager(cfg=cfg)
        return mgr.find_evolution_path(str(start_id), str(end_id))

    @mcp.tool()
    def citation_graph_query(
        action: str,
        doc_id: Optional[str] = None,
        ref_id: Optional[str] = None,
        top_k: int = 10,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Query local citation graph (references/citations/cocitation).

        Args:
            action: One of {"references","citations","cocitation"}.
            doc_id: Document id (required for references/cocitation).
            ref_id: Reference id (required for citations).
            top_k: For cocitation, number of related docs to return.
            limit: For references/citations, max rows.

        Returns:
            Dict with query results.
        """
        act = (action or "").strip().lower()
        store_cg = CitationGraphStore(db_path=default_citation_db_path(cfg))
        if act == "references":
            if not doc_id:
                raise ValueError("doc_id is required for action=references")
            return {"action": act, "doc_id": str(doc_id), "results": store_cg.get_references(str(doc_id), limit=int(limit))}
        if act == "citations":
            if not ref_id:
                raise ValueError("ref_id is required for action=citations")
            return {"action": act, "ref_id": str(ref_id), "results": store_cg.get_citations(str(ref_id), limit=int(limit))}
        if act == "cocitation":
            if not doc_id:
                raise ValueError("doc_id is required for action=cocitation")
            return {"action": act, "doc_id": str(doc_id), "top_k": int(top_k), "results": store_cg.get_cocitation(str(doc_id), top_k=int(top_k))}
        raise ValueError("Unsupported action. Use one of: references, citations, cocitation")

    @mcp.tool()
    def plan_research_request(
        prompt: str,
        *,
        mode: str = "standard",
        intent: str = "research",
    ) -> str:
        """Plan a research request as a replayable Markdown workflow (M1).

        This tool writes artifacts under ``output/workflows/<timestamp>/`` and returns Markdown
        suitable for an LLM to follow step-by-step.

        Args:
            prompt: User research request.
            mode: One of {"fast","standard"}. ``fast`` omits optional graph steps unless hinted.
            intent: High-level intent label (best-effort).

        Returns:
            Markdown text (not a large JSON payload).
        """
        m = (mode or "standard").strip().lower()
        if m not in {"fast", "standard"}:
            raise ValueError('mode must be one of: "fast", "standard"')

        plan, plan_md = build_and_write_workflow_plan_markdown(cfg=cfg, prompt=str(prompt), intent=str(intent), mode=m)  # type: ignore[arg-type]
        run_dir = plan_md.parent.resolve()
        run_jsonl = (run_dir / "run.jsonl").resolve()
        artifacts_dir = (run_dir / "artifacts").resolve()

        tools_lines: List[str] = []
        for st in plan.steps:
            tools_lines.append(f"- `{st.id}` → `{st.tool}`")

        artifacts_lines = [
            f"- plan: `{plan_md}`",
            f"- run log: `{run_jsonl}`",
            f"- artifacts dir: `{artifacts_dir}`",
            "",
            "Per-step suggested save paths (relative to the run directory):",
        ]
        for st in plan.steps:
            if (st.save_to or "").strip():
                artifacts_lines.append(f"- `{st.id}`: `{st.save_to}`")

        stop_lines: List[str] = []
        for n in plan.notes:
            stop_lines.append(f"- {n}")

        body = plan_md.read_text(encoding="utf-8").rstrip()
        extra = "\n".join(
            [
                "## Tools",
                "",
                "\n".join(tools_lines) if tools_lines else "- _No tools routed._",
                "",
                "## Artifacts",
                "",
                "\n".join(artifacts_lines),
                "",
                "## Stop conditions",
                "",
                "\n".join(stop_lines) if stop_lines else "- _No explicit stop conditions._",
                "",
            ]
        )
        return f"{body}\n{extra}"

    return mcp

