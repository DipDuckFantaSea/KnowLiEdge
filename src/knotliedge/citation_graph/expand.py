from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import quote, urlencode

import requests

from knotliedge.citation_graph.openalex_store import (
    OpenAlexCitationStore,
    OpenAlexCiteEdge,
    OpenAlexWorkRecord,
    normalize_openalex_work_id,
)
from knotliedge.citation_graph.hitl import OpenAlexExpansionStagingStore
from knotliedge.citation_graph.store import CitationGraphStore, default_citation_db_path, now_iso8601
from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.sources.openalex.inverted_abstract import restore_abstract_from_inverted_index
from knotliedge.storage.fts_store import default_fts_db_path

logger = setup_logging()

CHECKPOINT_KEY = "expand_citation_network_v1"
DEFAULT_MAX_DEPTH = 2
DEFAULT_PER_NODE_CITES = 30
DEFAULT_SLEEP_S = 0.12
DEFAULT_TIMEOUT_S = 35
DEFAULT_VERIFY_SSL = True


def _tail_from_work_url(wid: str) -> Optional[str]:
    n = normalize_openalex_work_id(wid)
    if not n:
        return None
    return n.rsplit("/", 1)[-1]


def _resolve_work_id_from_doi(
    *,
    session: requests.Session,
    doi: str,
    mailto: str,
    api_key: Optional[str],
    base_url: str,
    timeout_s: int,
    verify_ssl: bool,
) -> Optional[str]:
    d = (doi or "").strip()
    if not d:
        return None
    d = d.removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:").strip()
    if not d:
        return None
    params: Dict[str, object] = {"filter": f"doi:{d}", "per-page": 1, "mailto": mailto, "select": "id"}
    if (api_key or "").strip():
        params["api_key"] = str(api_key).strip()
    url = f"{base_url.rstrip('/')}/works?{urlencode(params)}"
    for attempt in range(3):
        try:
            r = session.get(url, timeout=timeout_s, verify=bool(verify_ssl))
            if r.status_code == 429 or (500 <= r.status_code < 600):
                time.sleep(1.0 * (2**attempt))
                continue
            if r.status_code != 200:
                logger.warning("OpenAlex DOI resolve non-200 | status=%s doi=%s", r.status_code, d)
                return None
            data = r.json() or {}
            if not isinstance(data, dict):
                return None
            results = data.get("results") or []
            if not isinstance(results, list) or not results:
                return None
            top = results[0]
            if not isinstance(top, dict):
                return None
            wid = top.get("id")
            return normalize_openalex_work_id(str(wid or "")) if isinstance(wid, str) else None
        except requests.exceptions.SSLError as e:
            if verify_ssl:
                logger.warning("OpenAlex SSL error on DOI resolve; retry verify=False | %s", e)
                verify_ssl = False
                time.sleep(0.5)
                continue
            time.sleep(1.0 * (2**attempt))
        except Exception as e:
            logger.warning("OpenAlex DOI resolve error | attempt=%s | doi=%s | %s", attempt, d, e)
            time.sleep(1.0 * (2**attempt))
    return None


def _venue_name(work: Dict[str, Any]) -> Optional[str]:
    pl = work.get("primary_location")
    if isinstance(pl, dict):
        src = pl.get("source")
        if isinstance(src, dict):
            dn = src.get("display_name")
            if isinstance(dn, str) and dn.strip():
                return dn.strip()
    hv = work.get("host_venue")
    if isinstance(hv, dict):
        dn = hv.get("display_name")
        if isinstance(dn, str) and dn.strip():
            return dn.strip()
    return None


def _authors_json(work: Dict[str, Any]) -> Optional[str]:
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return None
    names: List[str] = []
    for a in authorships:
        if not isinstance(a, dict):
            continue
        author = a.get("author")
        if isinstance(author, dict):
            dn = author.get("display_name")
            if isinstance(dn, str) and dn.strip():
                names.append(dn.strip())
    if not names:
        return None
    return json.dumps(names[:80], ensure_ascii=False)


def _doi_from_work(work: Dict[str, Any]) -> Optional[str]:
    d = work.get("doi")
    if isinstance(d, str) and d.strip():
        return d.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    ids = work.get("ids")
    if isinstance(ids, dict):
        dd = ids.get("doi")
        if isinstance(dd, str) and dd.strip():
            return dd.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    return None


def _fetch_work(
    *,
    session: requests.Session,
    work_id: str,
    mailto: str,
    api_key: Optional[str],
    base_url: str,
    timeout_s: int,
    verify_ssl: bool,
) -> Optional[Dict[str, Any]]:
    tail = _tail_from_work_url(work_id)
    if not tail:
        return None
    params: Dict[str, object] = {"mailto": mailto}
    if (api_key or "").strip():
        params["api_key"] = str(api_key).strip()
    url = f"{base_url.rstrip('/')}/works/{quote(tail, safe='')}?{urlencode(params)}"
    for attempt in range(4):
        try:
            r = session.get(url, timeout=timeout_s, verify=bool(verify_ssl))
            if r.status_code == 429 or (500 <= r.status_code < 600):
                wait = 1.5 * (2**attempt)
                logger.warning("OpenAlex backoff | status=%s | wait=%ss | url=%s", r.status_code, wait, url)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                logger.warning("OpenAlex work fetch non-200 | status=%s | url=%s", r.status_code, url)
                return None
            data = r.json()
            return data if isinstance(data, dict) else None
        except requests.exceptions.SSLError as e:
            if verify_ssl:
                logger.warning("OpenAlex SSL error; retry with verify=False | %s | url=%s", e, url)
                verify_ssl = False
                time.sleep(0.5)
                continue
            logger.warning("OpenAlex SSL error | attempt=%s | %s | url=%s", attempt, e, url)
            time.sleep(1.0 * (2**attempt))
        except Exception as e:
            logger.warning("OpenAlex work fetch error | attempt=%s | %s | url=%s", attempt, e, url)
            time.sleep(1.0 * (2**attempt))
    return None


def _fetch_citing_works(
    *,
    session: requests.Session,
    cited_tail: str,
    mailto: str,
    api_key: Optional[str],
    base_url: str,
    per_page: int,
    timeout_s: int,
    verify_ssl: bool,
) -> List[Dict[str, Any]]:
    params: Dict[str, object] = {
        "filter": f"cites:{cited_tail}",
        "per_page": int(per_page),
        "sort": "publication_year:desc",
        "mailto": mailto,
        "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index,primary_location,authorships",
    }
    if (api_key or "").strip():
        params["api_key"] = str(api_key).strip()
    url = f"{base_url.rstrip('/')}/works?{urlencode(params)}"
    for attempt in range(4):
        try:
            r = session.get(url, timeout=timeout_s, verify=bool(verify_ssl))
            if r.status_code == 429 or (500 <= r.status_code < 600):
                time.sleep(1.5 * (2**attempt))
                continue
            if r.status_code != 200:
                logger.warning("OpenAlex cites fetch non-200 | status=%s | url=%s", r.status_code, url)
                return []
            data = r.json() or {}
            if not isinstance(data, dict):
                return []
            results = data.get("results") or []
            if not isinstance(results, list):
                return []
            out: List[Dict[str, Any]] = []
            for it in results:
                if isinstance(it, dict):
                    out.append(it)
            return out
        except requests.exceptions.SSLError as e:
            if verify_ssl:
                logger.warning("OpenAlex SSL error; retry with verify=False | %s | url=%s", e, url)
                verify_ssl = False
                time.sleep(0.5)
                continue
            logger.warning("OpenAlex SSL error | attempt=%s | %s | url=%s", attempt, e, url)
            time.sleep(1.0 * (2**attempt))
        except Exception as e:
            logger.warning("OpenAlex cites fetch error | attempt=%s | %s", attempt, e)
            time.sleep(1.0 * (2**attempt))
    return []


def _work_to_record(
    work: Dict[str, Any],
    *,
    seed_kind: Optional[str],
    depth_seen: int,
    updated_at: str,
) -> Optional[OpenAlexWorkRecord]:
    wid = normalize_openalex_work_id(str(work.get("id") or ""))
    if not wid:
        return None
    inv = work.get("abstract_inverted_index")
    abstract = restore_abstract_from_inverted_index(inv) or None
    py = work.get("publication_year")
    yi: Optional[int] = int(py) if isinstance(py, int) else None
    cbc = work.get("cited_by_count")
    ci: Optional[int] = int(cbc) if isinstance(cbc, int) else None
    title = work.get("title")
    ts = str(title).strip() if isinstance(title, str) else None
    return OpenAlexWorkRecord(
        work_id=wid,
        doi=_doi_from_work(work),
        title=ts,
        publication_year=yi,
        host_venue_display_name=_venue_name(work),
        authors_json=_authors_json(work),
        abstract=abstract,
        cited_by_count=ci,
        seed_kind=seed_kind,
        depth_seen=int(depth_seen),
        updated_at=updated_at,
    )


def _collect_seeds_from_fts(fts_path: Path, *, limit: int) -> List[str]:
    if not fts_path.is_file():
        return []
    out: List[str] = []
    seen: Set[str] = set()
    try:
        con = sqlite3.connect(str(fts_path))
        con.row_factory = sqlite3.Row
        for row in con.execute(
            """
            SELECT DISTINCT openalex_id FROM documents
            WHERE openalex_id IS NOT NULL AND trim(openalex_id) != ''
            LIMIT ?;
            """,
            (int(limit),),
        ):
            raw = str(row["openalex_id"] or "").strip()
            n = normalize_openalex_work_id(raw)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        con.close()
    except Exception as e:
        logger.warning("FTS seed collection failed | path=%s | %s", fts_path, e)
    return out


def run_expand(
    *,
    config_path: Path,
    citation_db: Optional[Path],
    max_depth: int,
    per_node_cites: int,
    sleep_s: float,
    seed_limit: int,
    resume: bool,
    dry_run: bool,
    stage_only: bool = False,
    stage_run_id: Optional[str] = None,
    seed_work_ids: Optional[List[str]] = None,
) -> None:
    cfg = load_app_config(Path(config_path))
    mailto = (cfg.openalex.mailto or "").strip()
    if not mailto:
        raise ValueError("config openalex.mailto is required (OpenAlex Polite Pool).")

    cite_db = Path(citation_db) if citation_db is not None else default_citation_db_path(cfg)
    cg = CitationGraphStore(db_path=cite_db)
    oa = OpenAlexCitationStore(db_path=cite_db)
    stg = OpenAlexExpansionStagingStore(db_path=cite_db)
    run_id = str(stage_run_id or "").strip() or now_iso8601().replace(":", "").replace("-", "")

    dois_from_edges = cg.list_doi_reference_seeds(limit=seed_limit)
    fts_path = default_fts_db_path(cfg)
    fts_seeds = _collect_seeds_from_fts(fts_path, limit=seed_limit)
    extra_work_seeds: List[str] = []
    for x in seed_work_ids or []:
        n = normalize_openalex_work_id(str(x).strip())
        if n:
            extra_work_seeds.append(n)

    if dry_run:
        logger.info(
            "dry-run | doi_seeds=%s fts_openalex_seeds=%s extra_work_seeds=%s",
            len(dois_from_edges),
            len(fts_seeds),
            len(extra_work_seeds),
        )
        for d in dois_from_edges[:20]:
            logger.info("doi_seed | %s", d)
        for w in fts_seeds[:20]:
            logger.info("fts_seed | %s", w)
        for w in extra_work_seeds[:20]:
            logger.info("extra_work_seed | %s", w)
        return

    session = requests.Session()
    session.headers.update({"User-Agent": f"knotliedge-expand-citation/0.1 mailto:{mailto}"})
    api_key = (cfg.openalex.api_key or "").strip() or None
    base_url = "https://api.openalex.org"
    verify_ssl = bool(DEFAULT_VERIFY_SSL)

    seeds: List[str] = []
    for doi in dois_from_edges:
        wid = _resolve_work_id_from_doi(
            session=session,
            doi=doi,
            mailto=mailto,
            api_key=api_key,
            base_url=base_url,
            timeout_s=20,
            verify_ssl=verify_ssl,
        )
        if wid:
            seeds.append(wid)
        time.sleep(float(sleep_s))

    for wid in fts_seeds:
        if wid not in seeds:
            seeds.append(wid)
    for wid in extra_work_seeds:
        if wid not in seeds:
            seeds.append(wid)

    seeds = list(dict.fromkeys(seeds))
    logger.info("Collected %s seed work ids", len(seeds))

    done: Set[str] = set()
    q: Deque[Tuple[str, int, str]] = deque()
    if resume:
        blob = (stg.get_checkpoint(run_id=run_id, key=CHECKPOINT_KEY) if stage_only else oa.get_checkpoint(CHECKPOINT_KEY)) or {}
        raw_p = blob.get("processed")
        if isinstance(raw_p, list):
            for x in raw_p:
                if isinstance(x, str) and x.strip():
                    nw = normalize_openalex_work_id(x.strip())
                    if nw:
                        done.add(nw)
        raw_q = blob.get("queue")
        if isinstance(raw_q, list):
            for item in raw_q:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    w, d, k = item[0], item[1], item[2]
                    if isinstance(w, str) and isinstance(d, int) and isinstance(k, str):
                        nw = normalize_openalex_work_id(w)
                        if nw:
                            q.append((nw, d, k))
    for s in seeds:
        nw = normalize_openalex_work_id(s)
        if nw and nw not in done:
            q.append((nw, 0, "seed"))

    while q:
        work_id, depth, seed_kind = q.popleft()
        if work_id in done:
            continue
        if depth > int(max_depth):
            continue
        ts = now_iso8601()
        work = _fetch_work(
            session=session,
            work_id=work_id,
            mailto=mailto,
            api_key=api_key,
            base_url=base_url,
            timeout_s=DEFAULT_TIMEOUT_S,
            verify_ssl=verify_ssl,
        )
        time.sleep(float(sleep_s))
        if not work:
            logger.warning(
                "OpenAlex fetch returned empty; marking work done to avoid infinite re-queue | work_id=%s",
                work_id,
            )
            done.add(work_id)
            continue
        rec = _work_to_record(work, seed_kind=seed_kind, depth_seen=depth, updated_at=ts)
        if rec:
            if stage_only:
                stg.stage_work(run_id=run_id, rec=rec)
            else:
                oa.upsert_work(rec)
        done.add(work_id)

        refs = work.get("referenced_works") or []
        edges: List[OpenAlexCiteEdge] = []
        if isinstance(refs, list):
            for r in refs:
                if not isinstance(r, str):
                    continue
                child = normalize_openalex_work_id(r.strip())
                if not child:
                    continue
                edges.append(OpenAlexCiteEdge(src_work_id=work_id, dst_work_id=child, source="referenced_works", created_at=ts))
                if depth < int(max_depth):
                    q.append((child, depth + 1, "referenced"))

        if stage_only:
            stg.stage_edges(run_id=run_id, edges=edges)
        else:
            oa.upsert_edges(edges)

        if depth < int(max_depth):
            tail = _tail_from_work_url(work_id)
            if tail:
                citing = _fetch_citing_works(
                    session=session,
                    cited_tail=tail,
                    mailto=mailto,
                    api_key=api_key,
                    base_url=base_url,
                    per_page=int(per_node_cites),
                    timeout_s=DEFAULT_TIMEOUT_S,
                    verify_ssl=verify_ssl,
                )
                time.sleep(float(sleep_s))
                cite_edges: List[OpenAlexCiteEdge] = []
                for cw in citing:
                    cid = normalize_openalex_work_id(str(cw.get("id") or ""))
                    if not cid:
                        continue
                    cite_edges.append(OpenAlexCiteEdge(src_work_id=cid, dst_work_id=work_id, source="cites_filter", created_at=ts))
                    crec = _work_to_record(cw, seed_kind="citing", depth_seen=depth + 1, updated_at=ts)
                    if crec:
                        if stage_only:
                            stg.stage_work(run_id=run_id, rec=crec)
                        else:
                            oa.upsert_work(crec)
                    if depth + 1 <= int(max_depth):
                        q.append((cid, depth + 1, "citing"))
                if stage_only:
                    stg.stage_edges(run_id=run_id, edges=cite_edges)
                else:
                    oa.upsert_edges(cite_edges)

        if stage_only:
            stg.set_checkpoint(
                run_id=run_id,
                key=CHECKPOINT_KEY,
                payload={"version": 1, "processed": sorted(done), "queue": [[w, d, k] for w, d, k in q]},
            )
        else:
            oa.set_checkpoint(
                CHECKPOINT_KEY,
                {"version": 1, "processed": sorted(done), "queue": [[w, d, k] for w, d, k in q]},
            )

    if stage_only:
        c = stg.counts(run_id=run_id)
        logger.info(
            "Expansion staged | run_id=%s | pending_works=%s pending_edges=%s | db=%s",
            run_id,
            c.pending_works,
            c.pending_edges,
            cite_db,
        )
    else:
        logger.info("Expansion finished | done_works=%s | db=%s", len(done), cite_db)

