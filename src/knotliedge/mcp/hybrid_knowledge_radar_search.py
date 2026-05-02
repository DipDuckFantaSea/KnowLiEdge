from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from knotliedge.config.types import AppConfig
from knotliedge.retrieval.rrf import RankedId, rrf_merge, rrf_merge_rankings
from knotliedge.storage.chroma_store import ChromaStore
from knotliedge.storage.fts_store import FtsStore, default_fts_db_path
from knotliedge.llm.project_env import load_project_dotenv
from knotliedge.storage.venue_radar_store import VenueRadarStore, default_venue_radar_db_path

logger = logging.getLogger(__name__)


def format_md_block(*, title: str, lines: List[str]) -> str:
    body = "\n".join(lines).strip()
    if not body:
        body = "_无命中_"
    return f"{title}\n\n{body}\n"


def _strip_html_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


def _strip_markdown_images(s: str) -> str:
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s or "")
    return t


def _strip_inline_mathish(s: str) -> str:
    t = str(s or "")
    t = re.sub(r"\$[^$]{0,800}\$", " ", t)
    t = t.replace("$", " ")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = re.sub(r"\\", " ", t)
    return t


def one_line_preview(s: str, *, max_len: int, strip_images: bool = True) -> str:
    t = _strip_html_tags(str(s or ""))
    if strip_images:
        t = _strip_markdown_images(t)
    t = _strip_inline_mathish(t)
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n+", " ", t).strip()
    if not t:
        return ""
    if t.lstrip().startswith("#"):
        t = "\\" + t.lstrip()
    if len(t) > int(max_len):
        t = t[: int(max_len)].rstrip() + "…"
    return t


def sanitize_abstract_for_md(s: str, *, max_chars: int) -> str:
    t = _strip_html_tags(str(s or ""))
    t = _strip_markdown_images(t)
    t = _strip_inline_mathish(t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if int(max_chars) > 0 and len(t) > int(max_chars):
        t = t[: int(max_chars)].rstrip() + "…"
    return t


def fetch_documents_meta_map(cfg: AppConfig, doc_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
    ids = [str(x).strip() for x in doc_ids if str(x).strip()]
    if not ids:
        return {}
    db_path = default_fts_db_path(cfg)
    out: Dict[str, Dict[str, object]] = {}
    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
    except Exception as e:
        logger.warning("Failed to connect FTS sqlite for documents meta | path=%s | %s", db_path, e)
        return {}

    try:
        qmarks = ",".join("?" for _ in ids)
        sql = f"""
            SELECT doc_id, doi, openalex_id, citation_count, publication_year, journal_name,
                   openalex_title, openalex_authors_json
            FROM documents
            WHERE doc_id IN ({qmarks});
        """
        for row in con.execute(sql, ids):
            did = str(row["doc_id"] or "").strip()
            if not did:
                continue
            authors: List[str] = []
            raw_auth = row["openalex_authors_json"]
            if isinstance(raw_auth, str) and raw_auth.strip():
                try:
                    obj = json.loads(raw_auth)
                    if isinstance(obj, list):
                        authors = [str(a).strip() for a in obj if str(a).strip()]
                except Exception:
                    authors = []
            out[did] = {
                "doi": str(row["doi"] or ""),
                "openalex_id": str(row["openalex_id"] or ""),
                "citation_count": row["citation_count"],
                "publication_year": row["publication_year"],
                "journal_name": str(row["journal_name"] or ""),
                "openalex_title": str(row["openalex_title"] or ""),
                "authors": authors,
            }
    except Exception as e:
        logger.warning("Failed to read documents meta | path=%s | %s", db_path, e)
    finally:
        try:
            con.close()
        except Exception:
            pass
    return out


def read_frontmatter_fields(md_path: str) -> Dict[str, object]:
    p = Path(str(md_path or "")).expanduser()
    if not p.exists() or not p.is_file():
        return {}
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    if not raw.startswith("---"):
        return {}
    end = raw.find("\n---", 3)
    if end == -1:
        return {}
    block = raw[3:end]
    try:
        obj = yaml.safe_load(block) or {}
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, object] = {}
    title = obj.get("title")
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()
    venue = obj.get("venue")
    if isinstance(venue, str) and venue.strip():
        out["venue"] = venue.strip()
    year = obj.get("year")
    if isinstance(year, int):
        out["year"] = year
    elif isinstance(year, str) and year.strip().isdigit():
        out["year"] = int(year.strip())
    authors = obj.get("authors")
    if isinstance(authors, list):
        out["authors"] = [str(a).strip() for a in authors if str(a).strip()]
    sn = obj.get("short_name")
    if isinstance(sn, str) and sn.strip():
        out["short_name"] = sn.strip()
    return out


_MD_IMAGE_LINK = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")
_IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"})


def rewrite_local_image_paths_for_report(
    text: str,
    *,
    doc_short_name: str,
    cfg: AppConfig,
    report_md_dir: Path,
) -> str:
    """Rewrite ``![](...)`` so images resolve from ``report_md_dir`` (e.g. ``.../data/07_venue_radar``).

    Indexed previews often still contain MinerU-style ``images/<hash>.jpg`` or a mismatched
    ``assets/<short>/...`` prefix. Canonical files live under ``paths.markdown_assets_dir``;
    this emits a relative path such as ``../02_markdown_vault/assets/<short_name>/images/...``.
    """

    sn = (doc_short_name or "").strip()
    if not sn:
        return text or ""
    assets_root = Path(cfg.paths.markdown_assets_dir).resolve()
    out_dir = Path(report_md_dir).resolve()

    def repl(m: re.Match[str]) -> str:
        inner = m.group(2)
        path_only = inner.split('"', 1)[0].strip().split("?", 1)[0].strip()
        tail = inner[len(path_only) :]
        low = path_only.lower()
        if low.startswith(("http://", "https://", "data:")):
            return m.group(0)
        rel = path_only.lstrip("./").replace("\\", "/").lstrip("/")
        if not rel:
            return m.group(0)
        suf = Path(rel).suffix.lower()
        if suf not in _IMAGE_EXT:
            return m.group(0)

        rest: str
        am = re.match(r"^assets/[^/]+/(.+)$", rel)
        if am:
            rest = am.group(1)
        elif rel.startswith(("images/", "figures/")):
            rest = rel
        else:
            return m.group(0)

        target = (assets_root / sn / rest).resolve()
        try:
            rel_out = Path(os.path.relpath(target, start=out_dir)).as_posix()
        except ValueError:
            return m.group(0)
        return f"{m.group(1)}{rel_out}{tail}{m.group(3)}"

    return _MD_IMAGE_LINK.sub(repl, text or "")


def split_search_keywords(q: str) -> List[str]:
    """Split a user query into discrete keywords.

    - Double-quoted segments count as a single keyword (internal ``\"`` escaped per usual).
    - Outside quotes, split on whitespace, ``,,，、;；``, ``/／``, parentheses ``（）()``, and colons ``：:``.
    - De-duplicates case-insensitively while preserving first-seen order.

    Args:
        q: Raw query string.

    Returns:
        Non-empty keyword list; falls back to ``[q.strip()]`` when nothing can be parsed.
    """

    s = (q or "").strip()
    if not s:
        return []
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in " \t\n\r,，、;；/／（）():：":
            i += 1
            continue
        if ch == '"':
            j = s.find('"', i + 1)
            if j == -1:
                inner = s[i + 1 :].strip()
                if inner:
                    out.append(inner)
                break
            inner = s[i + 1 : j].strip()
            if inner:
                out.append(inner)
            i = j + 1
            continue
        j = i
        while j < n and s[j] not in " \t\n\r,，、;；/／（）():：\"":
            j += 1
        piece = s[i:j].strip()
        for _strip in (")", "）", "(", "（", "/", "／"):
            piece = piece.strip(_strip)
        piece = piece.strip()
        if piece:
            out.append(piece)
        i = j
    if not out:
        out = [s]
    seen: set[str] = set()
    deduped: List[str] = []
    for t in out:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


def _fts_quoted_term(t: str) -> str:
    inner = (t or "").strip().replace('"', '""')
    if not inner:
        return ""
    return f'"{inner}"'


def build_fts_or_query(keywords: Sequence[str]) -> str:
    """Build an FTS5 ``OR`` expression from multiple keywords (each term quoted)."""

    parts = [_fts_quoted_term(k) for k in keywords if (k or "").strip()]
    parts = [p for p in parts if p]
    if not parts:
        return (keywords[0] if keywords else "").strip()
    if len(parts) == 1:
        return parts[0]
    return " OR ".join(parts)


def _fts_search_robust(fts_store: FtsStore, fts_q: str, *, top_k: int) -> List[Any]:
    try:
        return fts_store.search(fts_q, top_k=int(top_k), where=None)
    except sqlite3.OperationalError:
        q_safe = re.sub(r"[-_/]+", " ", fts_q).strip()
        return fts_store.search(q_safe or fts_q, top_k=int(top_k), where=None)


def _ranked_docs_from_chunks(chunks: Sequence[Dict[str, Any]]) -> List[Tuple[str, float]]:
    best: Dict[str, float] = {}
    for c in chunks:
        if not isinstance(c, dict):
            continue
        did = str(c.get("doc_id") or "").strip()
        if not did:
            continue
        try:
            r = float(c.get("rrf_score") or 0.0)
        except Exception:
            r = 0.0
        best[did] = max(best.get(did, 0.0), r)
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def resolve_keyword_representatives(
    keywords: Sequence[str],
    per_kw_doc_ranks: Dict[str, List[Tuple[str, float]]],
) -> List[Dict[str, Any]]:
    """Assign one representative doc per keyword; resolve #1 collisions by max RRF.

    If the same ``doc_id`` is at the front of multiple keyword queues, the keyword
    with the highest RRF for that doc keeps it; the others pop and retry next round.
    """

    kws = list(keywords)
    queues: Dict[str, List[Tuple[str, float]]] = {kw: list(per_kw_doc_ranks.get(kw, [])) for kw in kws}
    assigned: Dict[str, Tuple[str, float]] = {}
    max_iters = sum(len(queues[kw]) for kw in kws) + len(kws) + 8
    for _ in range(max_iters):
        if len(assigned) >= len(kws):
            break
        heads: List[Tuple[str, str, float]] = []
        for kw in kws:
            if kw in assigned:
                continue
            if not queues[kw]:
                continue
            d, r = queues[kw][0]
            heads.append((kw, d, r))
        if not heads:
            break
        by_doc: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for kw, d, r in heads:
            by_doc[d].append((kw, r))
        winners: set[str] = set()
        for _d, lst in by_doc.items():
            if len(lst) == 1:
                kw, r = lst[0]
                assigned[kw] = (_d, r)
                winners.add(kw)
            else:
                kw_win, r_win = max(lst, key=lambda x: x[1])
                assigned[kw_win] = (_d, float(r_win))
                winners.add(kw_win)
                for kw, _r in lst:
                    if kw != kw_win and queues[kw] and queues[kw][0][0] == _d:
                        queues[kw].pop(0)
        for kw in winners:
            d_ass, _ = assigned[kw]
            if queues[kw] and queues[kw][0][0] == d_ass:
                queues[kw].pop(0)
    rows: List[Dict[str, Any]] = []
    for kw in kws:
        if kw not in assigned:
            continue
        d, r = assigned[kw]
        rows.append({"keyword": kw, "doc_id": d, "rrf": float(r)})
    rows.sort(key=lambda x: float(x["rrf"]), reverse=True)
    return rows


def _doc_stub_from_chunks(chunks: Sequence[Dict[str, Any]], did: str) -> Optional[Dict[str, object]]:
    rel = [c for c in chunks if isinstance(c, dict) and str(c.get("doc_id") or "").strip() == did]
    if not rel:
        return None
    rel_sorted = sorted(rel, key=lambda x: float(x.get("rrf_score") or 0.0), reverse=True)
    r0 = rel_sorted[0]
    try:
        best = max(float(c.get("rrf_score") or 0.0) for c in rel_sorted)
    except Exception:
        best = 0.0
    item: Dict[str, object] = {
        "doc_id": did,
        "short_name": str(r0.get("short_name") or ""),
        "source_md": str(r0.get("source_md") or ""),
        "best_rrf": float(best),
        "chunks": rel_sorted,
        "frontmatter": read_frontmatter_fields(str(r0.get("source_md") or "")),
    }
    return item


def _chunks_after_rrf(
    fts_store: FtsStore,
    store: ChromaStore,
    fts_q: str,
    vec_emb: Optional[Sequence[float]] = None,
    *,
    vec_embs: Optional[Sequence[Sequence[float]]] = None,
    vec_keyword_labels: Optional[Sequence[str]] = None,
    top_k_fts: int,
    top_k_vec: int,
    rrf_k: int,
    rrf_limit: int,
    keyword_note: Optional[str] = None,
) -> List[Dict[str, Any]]:
    fts_hits = _fts_search_robust(fts_store, fts_q, top_k=int(top_k_fts))
    fts_ranked = [RankedId(chunk_id=h.chunk_id, rank=i + 1, score=h.score) for i, h in enumerate(fts_hits)]
    fts_by_id = {h.chunk_id: h for h in fts_hits}

    multi_vec = vec_embs is not None and len(vec_embs) > 0
    if multi_vec:
        embs = list(vec_embs or [])
        vec_ranked_lists: List[List[RankedId]] = []
        vec_by_id: Dict[str, Any] = {}
        for emb in embs:
            vec_hits_i = store.search_by_embedding(emb, top_k=int(top_k_vec), where=None)
            vec_ranked_lists.append(
                [RankedId(chunk_id=h.chunk_id, rank=i + 1, score=h.score) for i, h in enumerate(vec_hits_i)]
            )
            for h in vec_hits_i:
                prev = vec_by_id.get(h.chunk_id)
                if prev is None or float(h.score) > float(prev.score):
                    vec_by_id[h.chunk_id] = h
        merged_items = rrf_merge_rankings(
            [fts_ranked, *vec_ranked_lists],
            rrf_k=int(rrf_k),
            limit=int(rrf_limit),
        )
    else:
        if vec_emb is None:
            raise ValueError("_chunks_after_rrf: pass vec_emb or non-empty vec_embs")
        vec_hits = store.search_by_embedding(vec_emb, top_k=int(top_k_vec), where=None)
        vec_ranked = [RankedId(chunk_id=h.chunk_id, rank=i + 1, score=h.score) for i, h in enumerate(vec_hits)]
        vec_by_id = {h.chunk_id: h for h in vec_hits}
        merged_items = rrf_merge(fts=fts_ranked, vec=vec_ranked, rrf_k=int(rrf_k), limit=int(rrf_limit))

    meta_ids = [m.chunk_id for m in merged_items]
    meta_map = fts_store.get_chunk_meta_map(meta_ids)

    labels = list(vec_keyword_labels) if vec_keyword_labels is not None else []
    label_ok = bool(
        multi_vec
        and labels
        and vec_embs is not None
        and len(labels) == len(vec_embs)
    )

    out_results: List[Dict[str, Any]] = []
    for m in merged_items:
        cid = m.chunk_id
        vh = vec_by_id.get(cid)
        fh = fts_by_id.get(cid)
        meta = meta_map.get(cid, {})
        if multi_vec:
            ranks_t = m.ranks
            scores_t = m.scores
            fts_r = ranks_t[0] if len(ranks_t) > 0 else None
            vec_kw_ranks = list(ranks_t[1:]) if len(ranks_t) > 1 else []
            vec_kw_scores = list(scores_t[1:]) if len(scores_t) > 1 else []
            vec_ranks_present = [x for x in vec_kw_ranks if x is not None]
            vec_scores_present = [x for x in vec_kw_scores if x is not None]
            vec_rank_disp = int(min(vec_ranks_present)) if vec_ranks_present else None
            vec_score_disp = float(max(vec_scores_present)) if vec_scores_present else None
            fts_s = scores_t[0] if len(scores_t) > 0 else None
            row: Dict[str, Any] = {
                "chunk_id": cid,
                "rrf_score": float(m.rrf_score),
                "fts_rank": fts_r,
                "vec_rank": vec_rank_disp,
                "fts_score": fts_s,
                "vec_score": vec_score_disp,
                "doc_id": (vh.doc_id if vh else (fh.doc_id if fh else "")),
                "short_name": (vh.short_name if vh else (fh.short_name if fh else "")),
                "section": (vh.section if vh else (fh.section if fh else None)),
                "preview": (vh.preview if vh else (fh.preview if fh else "")),
                "source_md": (vh.source_md if vh else str(meta.get("source_md") or "")),
                "chunk_index": (int(meta.get("chunk_index") or 0) if meta else None),
            }
            if label_ok:
                row["vec_rank_by_keyword"] = {
                    labels[i]: vec_kw_ranks[i] for i in range(min(len(labels), len(vec_kw_ranks)))
                }
                row["vec_score_by_keyword"] = {
                    labels[i]: vec_kw_scores[i] for i in range(min(len(labels), len(vec_kw_scores)))
                }
        else:
            row = {
                "chunk_id": cid,
                "rrf_score": float(m.rrf_score),
                "fts_rank": m.fts_rank,
                "vec_rank": m.vec_rank,
                "fts_score": m.fts_score,
                "vec_score": m.vec_score,
                "doc_id": (vh.doc_id if vh else (fh.doc_id if fh else "")),
                "short_name": (vh.short_name if vh else (fh.short_name if fh else "")),
                "section": (vh.section if vh else (fh.section if fh else None)),
                "preview": (vh.preview if vh else (fh.preview if fh else "")),
                "source_md": (vh.source_md if vh else str(meta.get("source_md") or "")),
                "chunk_index": (int(meta.get("chunk_index") or 0) if meta else None),
            }
        if keyword_note:
            row["matched_keyword"] = keyword_note
        out_results.append(row)
    return out_results


def run_hybrid_knowledge_and_radar_search(
    cfg: AppConfig,
    store: ChromaStore,
    embed_query: Callable[[str], Sequence[float]],
    query: str,
    *,
    top_k_total: int = 20,
    top_k_local_docs: int = 10,
    top_k_radar_works: int = 10,
    local_chunks_per_doc: int = 3,
    top_k_fts: int = 80,
    top_k_vec: int = 80,
    rrf_k: int = 60,
    report_markdown_dir: Optional[Path] = None,
    ai_parse_query: bool = True,
) -> Dict[str, Any]:
    q0 = (query or "").strip()
    if not q0:
        return {"query": query, "markdown": "### 本地深度解析库命中\n\n_空查询_\n\n### 前沿雷达最新捕获\n\n_空查询_\n"}

    report_dir = (report_markdown_dir or default_venue_radar_db_path(cfg).parent).resolve()

    load_project_dotenv(cfg.project_root)

    # Translate to English first (for FTS+vector) if QUICK_TRANSLATE_API_KEY is configured.
    q = q0
    q_en: Optional[str] = None
    try:
        from knotliedge.llm.quick_translate import translate_query_to_english

        q_en = translate_query_to_english(project_root=cfg.project_root, query=q0, timeout_s=60.0)
        if q_en:
            q = q_en
    except Exception:
        q_en = None

    q_emb = embed_query(q)
    total = max(1, int(top_k_total))
    local_docs = max(0, int(top_k_local_docs))
    radar_works = max(0, int(top_k_radar_works))
    if local_docs + radar_works != total:
        local_docs = total // 2
        radar_works = total - local_docs
    per_doc = max(1, int(local_chunks_per_doc))
    rrf_limit = max(200, local_docs * per_doc * 4)

    def run_local() -> Dict[str, Any]:
        fts_store = FtsStore(db_path=default_fts_db_path(cfg))
        keyword_source = "heuristic"
        keywords: List[str] = []
        if ai_parse_query:
            from knotliedge.llm.query_keyword_expand import try_expand_query_to_keywords

            expanded = try_expand_query_to_keywords(project_root=cfg.project_root, query=q, timeout_s=120.0)
            if expanded:
                keywords = expanded
                keyword_source = "openai_template"
        if not keywords:
            keywords = split_search_keywords(q)
            keyword_source = "heuristic"
        multi = len(keywords) >= 2

        if not multi:
            chunks = _chunks_after_rrf(
                fts_store,
                store,
                q.strip(),
                q_emb,
                top_k_fts=int(top_k_fts),
                top_k_vec=int(top_k_vec),
                rrf_k=int(rrf_k),
                rrf_limit=int(rrf_limit),
                keyword_note=None,
            )
            per_kw: Dict[str, Dict[str, Any]] = {keywords[0]: {"chunks": chunks}}
            reps: List[Dict[str, Any]] = []
        else:
            fts_or = build_fts_or_query(keywords)
            kw_emb_map: Dict[str, Sequence[float]] = {kw: embed_query(kw) for kw in keywords}

            # Use full-query embedding for the primary FTS∪vec merge (one Chroma query).
            # Multi-embedding RRF in one call triggers intermittent access violations on
            # Windows Chroma Rust; per-keyword Chroma runs below remain single-query each.
            chunks = _chunks_after_rrf(
                fts_store,
                store,
                fts_or,
                q_emb,
                top_k_fts=int(top_k_fts),
                top_k_vec=int(top_k_vec),
                rrf_k=int(rrf_k),
                rrf_limit=int(rrf_limit),
                keyword_note=None,
            )
            per_kw = {}

            def one_kw(kw: str) -> Tuple[str, List[Dict[str, Any]]]:
                emb_kw = kw_emb_map[kw]
                ch = _chunks_after_rrf(
                    fts_store,
                    store,
                    kw.strip(),
                    emb_kw,
                    top_k_fts=int(top_k_fts),
                    top_k_vec=int(top_k_vec),
                    rrf_k=int(rrf_k),
                    rrf_limit=int(rrf_limit),
                    keyword_note=kw,
                )
                return kw, ch

            for kw in keywords:
                _kw, ch = one_kw(kw)
                per_kw[_kw] = {"chunks": ch}
            per_kw_rank = {kw: _ranked_docs_from_chunks((per_kw.get(kw) or {}).get("chunks") or []) for kw in keywords}
            reps = resolve_keyword_representatives(keywords, per_kw_rank)

        return {
            "chunks": chunks,
            "keywords": keywords,
            "keyword_source": keyword_source,
            "multi_keyword_mode": multi,
            "per_keyword": per_kw,
            "keyword_representatives": reps,
        }

    def run_radar() -> Dict[str, Any]:
        radar_store = ChromaStore(cfg=cfg, embedder=None, collection_name="openalex_abstracts")
        hits = radar_store.search_by_embedding(q_emb, top_k=int(radar_works), where=None)
        radar_db = VenueRadarStore(db_path=default_venue_radar_db_path(cfg))
        row_map = radar_db.get_abstracts_by_ids([h.chunk_id for h in hits])
        out: List[Dict[str, Any]] = []
        for h in hits:
            row = row_map.get(h.chunk_id, {})
            title = str(row.get("title") or "")
            abstract = str(row.get("abstract") or "")
            pub_date = str(row.get("publication_date") or "")
            venue_name = str(row.get("venue_name") or "")
            url = str(row.get("url") or "")
            authors = row.get("authors") if isinstance(row, dict) else []
            institutions = row.get("institutions") if isinstance(row, dict) else []
            in_local = bool(row.get("in_local_vault") or False) if isinstance(row, dict) else False
            local_doc_id = str(row.get("local_doc_id") or "") if isinstance(row, dict) else ""
            local_short_name = str(row.get("local_short_name") or "") if isinstance(row, dict) else ""
            local_md_path = str(row.get("local_md_path") or "") if isinstance(row, dict) else ""
            local_match = str(row.get("local_match") or "") if isinstance(row, dict) else ""
            out.append(
                {
                    "id": h.chunk_id,
                    "score": float(h.score),
                    "title": title,
                    "publication_date": pub_date,
                    "venue_name": venue_name,
                    "url": url,
                    "authors": authors if isinstance(authors, list) else [],
                    "institutions": institutions if isinstance(institutions, list) else [],
                    "abstract": abstract,
                    "in_local_vault": in_local,
                    "local_doc_id": local_doc_id,
                    "local_short_name": local_short_name,
                    "local_md_path": local_md_path,
                    "local_match": local_match,
                }
            )
        return {"results": out}

    # Chroma's Rust client is not safe for concurrent queries on one process; run serially.
    local = run_local()
    radar = run_radar()

    doc_map: Dict[str, Dict[str, object]] = {}
    combined_chunks = local.get("chunks") or []
    for r in combined_chunks:
        if not isinstance(r, dict):
            continue
        did = str(r.get("doc_id") or "").strip()
        if not did:
            continue
        item = doc_map.get(did)
        if item is None:
            item = {
                "doc_id": did,
                "short_name": str(r.get("short_name") or ""),
                "source_md": str(r.get("source_md") or ""),
                "best_rrf": float(r.get("rrf_score") or 0.0),
                "chunks": [],
            }
            fm0 = read_frontmatter_fields(str(item["source_md"]))
            item["frontmatter"] = fm0
            doc_map[did] = item
        else:
            try:
                item["best_rrf"] = max(float(item.get("best_rrf") or 0.0), float(r.get("rrf_score") or 0.0))
            except Exception:
                pass
        ch_list = item.get("chunks")
        if isinstance(ch_list, list):
            ch_list.append(r)

    multi_mode = bool(local.get("multi_keyword_mode"))
    reps = local.get("keyword_representatives") or []
    per_kw = local.get("per_keyword") or {}

    if multi_mode and isinstance(reps, list) and reps:
        ordered_ids: List[str] = []
        for row in reps:
            if not isinstance(row, dict):
                continue
            did = str(row.get("doc_id") or "").strip()
            if did and did not in ordered_ids:
                ordered_ids.append(did)
        for did, _ in _ranked_docs_from_chunks(combined_chunks):
            if len(ordered_ids) >= int(local_docs):
                break
            if did not in ordered_ids:
                ordered_ids.append(did)
        rep_kw_by_doc = {
            str(r.get("doc_id") or "").strip(): str(r.get("keyword") or "")
            for r in reps
            if isinstance(r, dict) and str(r.get("doc_id") or "").strip()
        }
        for did in ordered_ids:
            if did in doc_map:
                continue
            kw_own = rep_kw_by_doc.get(did, "")
            stub: Optional[Dict[str, object]] = None
            if kw_own:
                stub = _doc_stub_from_chunks((per_kw.get(kw_own) or {}).get("chunks") or [], did)
            if stub is None:
                for kw, pack in per_kw.items():
                    stub = _doc_stub_from_chunks((pack or {}).get("chunks") or [], did)
                    if stub is not None:
                        break
            if stub is not None:
                doc_map[did] = stub
        docs_sorted = [doc_map[did] for did in ordered_ids[: int(local_docs)] if did in doc_map]
    else:
        docs_sorted = sorted(doc_map.values(), key=lambda x: float(x.get("best_rrf") or 0.0), reverse=True)[: int(local_docs)]

    doc_ids_for_meta = [str(d.get("doc_id") or "").strip() for d in docs_sorted if str(d.get("doc_id") or "").strip()]
    oa_meta_by_doc = fetch_documents_meta_map(cfg, doc_ids_for_meta)

    local_lines: List[str] = []
    ks = str((local.get("keyword_source") or "")).strip()
    if ks:
        kws = local.get("keywords") if isinstance(local.get("keywords"), list) else []
        kws_s = " ".join(str(x) for x in kws if str(x).strip())
        local_lines.append(f"- **keyword_source**: `{ks}`  **keywords**: {kws_s or '_empty_'}")
        local_lines.append("")

    if multi_mode and isinstance(reps, list) and reps:
        local_lines.append("#### 关键词代表文献（多关键词 RRF 消解）")
        for row in reps:
            if not isinstance(row, dict):
                continue
            kw = str(row.get("keyword") or "").strip()
            did = str(row.get("doc_id") or "").strip()
            rrf_v = row.get("rrf")
            ditem = doc_map.get(did) if did else None
            title_s = ""
            if isinstance(ditem, dict):
                fm0 = ditem.get("frontmatter") if isinstance(ditem.get("frontmatter"), dict) else {}
                title_s = str(fm0.get("title") or "").strip()
            if not title_s and isinstance(ditem, dict):
                title_s = str(ditem.get("short_name") or "").strip()
            label = title_s or (did if did else "N/A")
            local_lines.append(f"- **{kw}** → {label}  (doc_id=`{did}`  RRF={rrf_v})")
        local_lines.append("")

    for i, d in enumerate(docs_sorted, start=1):
        fm = d.get("frontmatter") if isinstance(d.get("frontmatter"), dict) else {}
        did = str(d.get("doc_id") or "").strip()
        oa = oa_meta_by_doc.get(did, {}) if did else {}

        title = str(fm.get("title") or "").strip()
        if not title:
            title = str(oa.get("openalex_title") or "").strip()

        venue = str(fm.get("venue") or "").strip()
        if not venue:
            venue = str(oa.get("journal_name") or "").strip()

        year = fm.get("year")
        if year is None and oa.get("publication_year") is not None:
            try:
                year = int(oa.get("publication_year"))  # type: ignore[arg-type]
            except Exception:
                year = None

        authors = fm.get("authors") if isinstance(fm.get("authors"), list) else []
        if (not authors) and isinstance(oa.get("authors"), list):
            authors = [str(a).strip() for a in oa.get("authors") or [] if str(a).strip()]  # type: ignore[assignment]
        authors_s = ", ".join([str(a) for a in authors[:10]]) if authors else ""
        heading = title if title else str(d.get("short_name") or "")
        meta_bits = " | ".join([x for x in [str(year) if year else "", venue, authors_s] if x])
        local_lines.append(f"- **#{i}** {heading}" + (f"  \n  _{meta_bits}_" if meta_bits else ""))
        local_lines.append(f"  - **doc_id**: `{d.get('doc_id','')}`  **best_rrf**: {d.get('best_rrf')}")
        local_lines.append(f"  - **source_md**: `{d.get('source_md','')}`")
        chunks = d.get("chunks") if isinstance(d.get("chunks"), list) else []
        chunks = sorted(chunks, key=lambda x: float(x.get("rrf_score") or 0.0), reverse=True)[:per_doc]
        fm_doc = d.get("frontmatter") if isinstance(d.get("frontmatter"), dict) else {}
        doc_sn_base = str(d.get("short_name") or "").strip() or str(fm_doc.get("short_name") or "").strip()
        for j, c in enumerate(chunks, start=1):
            raw_prev = str(c.get("preview") or "")
            doc_sn = str(c.get("short_name") or "").strip() or doc_sn_base
            prev_src = rewrite_local_image_paths_for_report(
                raw_prev, doc_short_name=doc_sn, cfg=cfg, report_md_dir=report_dir
            )
            prev = one_line_preview(prev_src, max_len=360, strip_images=False) or "N/A"
            local_lines.append(
                "  - **hit_{j}**: `{cid}`  (RRF={rrf})  **section**: {sec}\n"
                "    - {prev}".format(
                    j=j,
                    cid=str(c.get("chunk_id") or ""),
                    rrf=c.get("rrf_score"),
                    sec=str(c.get("section") or "N/A"),
                    prev=prev,
                )
            )

    radar_lines: List[str] = []
    for i, r in enumerate(radar.get("results") or [], start=1):
        authors = r.get("authors") if isinstance(r, dict) else []
        insts = r.get("institutions") if isinstance(r, dict) else []
        authors_s = ", ".join([str(a) for a in authors[:10]]) if isinstance(authors, list) and authors else ""
        insts_s = ", ".join([str(a) for a in insts[:8]]) if isinstance(insts, list) and insts else ""
        abstract_s = sanitize_abstract_for_md(str(r.get("abstract") or ""), max_chars=6000)
        abstract_block = ""
        if abstract_s:
            abstract_block = "\n".join(["    " + ln for ln in abstract_s.splitlines()])
        local_tag = ""
        if bool(r.get("in_local_vault") or False):
            tag = "✅ 已在本地库收录"
            did = str(r.get("local_doc_id") or "").strip()
            sn = str(r.get("local_short_name") or "").strip()
            why = str(r.get("local_match") or "").strip()
            extra = []
            if did:
                extra.append(f"doc_id={did}")
            if sn:
                extra.append(f"short_name={sn}")
            if why:
                extra.append(f"match={why}")
            if extra:
                tag += "（" + ", ".join(extra) + "）"
            local_tag = f"  - **local**: {tag}"
        radar_lines.append(
            "\n".join(
                [
                    f"- **#{i}** `{r.get('id','')}`  (score={r.get('score')})",
                    f"  - **title**: {str(r.get('title') or '').strip() or 'N/A'}",
                    f"  - **venue**: {str(r.get('venue_name') or '').strip() or 'N/A'}  **date**: {str(r.get('publication_date') or '').strip() or 'N/A'}",
                    f"  - **authors**: {authors_s or 'N/A'}",
                    f"  - **institutions**: {insts_s or 'N/A'}",
                    f"  - **url**: `{str(r.get('url') or '').strip() or 'N/A'}`",
                    *( [local_tag] if local_tag else [] ),
                    *(["  - **abstract**:\n" + abstract_block] if abstract_block else ["  - **abstract**: N/A"]),
                ]
            )
        )

    md = "\n\n".join(
        [
            format_md_block(title="### 本地深度解析库命中", lines=local_lines),
            format_md_block(title="### 前沿雷达最新捕获", lines=radar_lines),
        ]
    ).strip() + "\n"

    return {
        "query": query,
        "top_k_total": int(total),
        "top_k_local_docs": int(local_docs),
        "top_k_radar_works": int(radar_works),
        "ai_parse_query": bool(ai_parse_query),
        "markdown": md,
        "local": local,
        "radar": radar,
    }
