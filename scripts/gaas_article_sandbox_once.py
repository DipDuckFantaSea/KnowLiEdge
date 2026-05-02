"""One-shot: PDF -> MD via mineru-api, then fused KB search + numbered citations (sandbox).

Run from project root:
  conda run -n agent python scripts/gaas_article_sandbox_once.py

Re-run annotation only (reuse existing converted MD):
  set KNOTLIEDGE_ANNOTATE_ONLY=1
  conda run -n agent python scripts/gaas_article_sandbox_once.py
"""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from knotliedge.config.load import load_app_config
from knotliedge.embeddings.lazy_ipc_gate import LazyIpcEmbedderSession
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.mineru.service import start_service, stop_service
from knotliedge.pipeline.watch_ingest import _parse_pdf_via_mineru_api
from knotliedge.retrieval.rrf import RankedId, rrf_merge
from knotliedge.storage.chroma_store import ChromaStore
from knotliedge.storage.fts_store import FtsStore, default_fts_db_path

logger = setup_logging()


def _fts_safe_query(text: str, *, max_len: int = 500) -> str:
    """Build an FTS5-safe query: tokenize alnum + CJK only (drops '.' ':' etc.)."""
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text)
    t = " ".join(tokens).strip()
    return t[:max_len]


ROOT = Path(__file__).resolve().parents[1]
SANDBOX_CFG = ROOT / "sandbox" / "configs" / "sandbox.yaml"
PDF_PATH = ROOT / "20260419_GaAs_lcx_v1.pdf"
OUT_DIR = ROOT / "sandbox" / "data" / "article_workspace" / "20260419_GaAs_lcx_v1"
OUT_MD = OUT_DIR / "20260419_GaAs_lcx_v1.md"
DOC_ID_PARSE = "article_gaas_20260419"


def _fused_search(
    *,
    cfg: Any,
    store: ChromaStore,
    embed_gate: LazyIpcEmbedderSession,
    query: str,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    embed_gate.require_for_store(store)
    fts_store = FtsStore(db_path=default_fts_db_path(cfg))
    try:
        fts_hits = fts_store.search(q, top_k=20, where=None)
    except Exception as e:
        logger.warning("FTS search failed, vector-only fallback: %s", e)
        fts_hits = []
    fts_ranked = [RankedId(chunk_id=h.chunk_id, rank=i + 1, score=h.score) for i, h in enumerate(fts_hits)]
    vec_embedder = embed_gate.get_optional()
    vec_hits = []
    vec_ranked: List[RankedId] = []
    if vec_embedder is not None:
        store.bind_embedder(vec_embedder)
        vec_hits = store.search(q, top_k=20, where=None)
        vec_ranked = [RankedId(chunk_id=h.chunk_id, rank=i + 1, score=h.score) for i, h in enumerate(vec_hits)]
    merged = rrf_merge(fts=fts_ranked, vec=vec_ranked, rrf_k=60, limit=int(top_k))
    vec_by_id = {h.chunk_id: h for h in vec_hits}
    fts_by_id = {h.chunk_id: h for h in fts_hits}
    meta_map = fts_store.get_chunk_meta_map([m.chunk_id for m in merged])
    out: List[Dict[str, Any]] = []
    for m in merged:
        cid = m.chunk_id
        vh = vec_by_id.get(cid)
        fh = fts_by_id.get(cid)
        meta = meta_map.get(cid, {})
        out.append(
            {
                "chunk_id": cid,
                "doc_id": (vh.doc_id if vh else (fh.doc_id if fh else "")),
                "short_name": (vh.short_name if vh else (fh.short_name if fh else "")),
                "preview": (vh.preview if vh else (fh.preview if fh else "")),
                "source_md": (vh.source_md if vh else str(meta.get("source_md") or "")),
            }
        )
    return out


def _unique_refs(
    rows: List[Dict[str, Any]],
    seen_docs: set[str],
    out_list: List[Dict[str, str]],
    *,
    max_new: int = 5,
) -> List[str]:
    """Append new bibliography entries; return new citation numbers as strings for this row batch."""
    nums: List[str] = []
    for r in rows:
        if len(nums) >= int(max_new):
            break
        did = str(r.get("doc_id") or "").strip()
        if not did or did in seen_docs:
            continue
        seen_docs.add(did)
        n = len(out_list) + 1
        label = f"S{n}"
        nums.append(label)
        out_list.append(
            {
                "n": label,
                "doc_id": did,
                "short_name": str(r.get("short_name") or ""),
                "source_md": str(r.get("source_md") or ""),
                "preview": (str(r.get("preview") or ""))[:240].replace("\n", " "),
            }
        )
    return nums


def _split_md_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown into (heading_line, section_body) using ATX headings at level 1–3."""
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    cur_h = ""
    cur_body: List[str] = []
    for line in lines:
        if re.match(r"^#{1,3}\s+\S", line):
            if cur_h or cur_body:
                sections.append((cur_h, "\n".join(cur_body).strip()))
            cur_h = line
            cur_body = []
        else:
            cur_body.append(line)
    if cur_h or cur_body:
        sections.append((cur_h, "\n".join(cur_body).strip()))
    return sections if sections else [("", text.strip())]


def _strip_previous_annotation(body: str) -> str:
    """Remove prior run's blockquotes and bibliography so re-annotate is idempotent."""
    for sep in ("## 参考文献（沙盒知识库", "## 参考文献（沙盒库"):
        if sep in body:
            body = body.split(sep)[0].rstrip()
            break
    out: List[str] = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("> 沙盒知识库") or s.startswith("> **〔沙盒知识库补充〕**"):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _annotate_body(body: str, cfg: Any, store: ChromaStore, embed_gate: LazyIpcEmbedderSession) -> Tuple[str, List[Dict[str, str]]]:
    """After each ATX section, insert blockquote with sandbox-only [Sn] cites (distinct from IEEE [n] in body)."""
    body = _strip_previous_annotation(body)
    bib: List[Dict[str, str]] = []
    seen_docs: set[str] = set()

    parts: List[str] = []
    for heading, sec in _split_md_sections(body):
        block = []
        if heading:
            block.append(heading)
        if sec:
            block.append(sec)
        sec_text = "\n".join(block).strip()
        if not sec_text:
            continue
        parts.append(sec_text)
        q = _fts_safe_query(sec_text[:1200])
        if len(q) < 12:
            q = "GaAs gallium arsenide semiconductor"
        hits = _fused_search(cfg=cfg, store=store, embed_gate=embed_gate, query=q, top_k=8)
        new_nums = _unique_refs(hits, seen_docs, bib, max_new=5)
        if new_nums:
            cite = "".join(f"[{n}]" for n in new_nums)
            parts.append("")
            parts.append(
                f"> **〔沙盒知识库补充〕**（与正文 IEEE 参考文献编号无关）{cite}"
            )
        parts.append("")

    if not parts:
        parts = [body.strip()]

    bib_lines = [
        "",
        "## 参考文献（沙盒知识库 knotliedge_v1_sandbox）",
        "",
        "以下编号为 **沙盒库专用**（形如 `[S1]`），**不同于**正文中 IEEE 格式的参考文献编号。",
        "",
    ]
    for b in bib:
        bib_lines.append(
            f"[{b['n']}] doc_id=`{b['doc_id']}` short_name=`{b['short_name']}` source_md=`{b['source_md']}`"
        )
        if b.get("preview"):
            bib_lines.append(f"    摘要片段：{b['preview']}…")
        bib_lines.append("")

    return "\n".join(parts).rstrip() + "\n".join(bib_lines), bib


def main() -> None:
    annotate_only = os.environ.get("KNOTLIEDGE_ANNOTATE_ONLY", "").strip() in {"1", "true", "yes"}
    if not annotate_only and not PDF_PATH.is_file():
        raise SystemExit(f"PDF not found: {PDF_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_app_config(SANDBOX_CFG)
    embed_gate = LazyIpcEmbedderSession(config_path=SANDBOX_CFG)
    store = ChromaStore(cfg=cfg, embedder=None)

    if annotate_only:
        if not OUT_MD.is_file():
            raise SystemExit(f"KNOTLIEDGE_ANNOTATE_ONLY set but MD missing: {OUT_MD}")
        md_text = OUT_MD.read_text(encoding="utf-8")
        logger.info("Annotate-only mode, loaded %s (%s chars)", OUT_MD, len(md_text))
    else:
        st = start_service(
            enable_vlm_preload=False,
            env_overrides={
                "MINERU_MODEL_SOURCE": "local",
                "MINERU_VIRTUAL_VRAM_SIZE": "8",
            },
            log_dir=(cfg.project_root / ".knotliedge"),
            work_dir=(cfg.project_root / ".knotliedge" / "mineru_api_workdir"),
        )
        api_url = st.api_url
        logger.info("mineru-api at %s", api_url)

        try:
            md_text, _images = _parse_pdf_via_mineru_api(
                cfg=cfg,
                api_url=api_url,
                pdf_path=PDF_PATH,
                doc_id=DOC_ID_PARSE,
                backend="pipeline",
                parse_method="txt",
                formula_enable=True,
                table_enable=True,
                return_images=False,
                timeout_s=900,
            )
        finally:
            stop_service()
            time.sleep(1.0)

        OUT_MD.write_text(md_text, encoding="utf-8")
        logger.info("Wrote raw markdown: %s", OUT_MD)
        out_base = cfg.project_root / ".knotliedge" / "mineru_api_workdir" / "output"
        if out_base.is_dir():
            candidates = sorted(
                out_base.glob(f"*/{DOC_ID_PARSE}/txt/images"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                img_dst = OUT_DIR / "images"
                shutil.copytree(candidates[0], img_dst, dirs_exist_ok=True)
                logger.info("Copied MinerU images -> %s", img_dst)

    annotated, _bib = _annotate_body(md_text, cfg, store, embed_gate)
    OUT_MD.write_text(annotated, encoding="utf-8")
    logger.info("Wrote annotated markdown: %s", OUT_MD)


if __name__ == "__main__":
    main()
