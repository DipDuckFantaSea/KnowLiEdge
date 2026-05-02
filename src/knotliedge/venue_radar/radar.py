from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlencode

from knotliedge.config.load import load_app_config
from knotliedge.embeddings import get_embedder
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.metadata.local_vault_presence import build_radar_local_match_map
from knotliedge.sources.openalex.inverted_abstract import restore_abstract_from_inverted_index
from knotliedge.storage.chroma_store import ChromaStore
from knotliedge.storage.venue_radar_store import VenueRadarStore, default_venue_radar_db_path

logger = setup_logging()


def _extract_source_id_for_filter(raw_source_id: str) -> Optional[str]:
    s = (raw_source_id or "").strip()
    if not s:
        return None
    if s.startswith("https://openalex.org/"):
        s = s[len("https://openalex.org/") :]
    s = s.strip()
    if s.startswith("S") and len(s) > 1:
        return s
    return None


def iter_openalex_works_by_venue(
    *,
    venue_source_id: str,
    from_publication_date: str,
    mailto: str,
    api_key: Optional[str],
    per_page: int = 200,
    base_url: str = "https://api.openalex.org",
    timeout_s: int = 30,
) -> Iterator[Dict[str, Any]]:
    """Iterate OpenAlex works by venue source id with cursor pagination."""
    try:
        import requests
    except Exception as e:
        raise RuntimeError(f"requests not available for OpenAlex radar fetch | {e}") from e

    src = _extract_source_id_for_filter(venue_source_id)
    if src is None:
        raise ValueError(f"Invalid OpenAlex source id: {venue_source_id}")
    mail = (mailto or "").strip()
    if not mail:
        raise ValueError("openalex.mailto is required (Polite Pool)")

    cursor = "*"
    while True:
        flt = f"primary_location.source.id:{src},from_publication_date:{from_publication_date}"
        params: Dict[str, object] = {
            "filter": flt,
            "per_page": int(per_page),
            "cursor": cursor,
            "mailto": mail,
            "select": ",".join(
                [
                    "id",
                    "title",
                    "publication_date",
                    "primary_location",
                    "abstract_inverted_index",
                    "authorships",
                    "doi",
                    "ids",
                ]
            ),
        }
        k = (api_key or "").strip()
        if k:
            params["api_key"] = k

        url = f"{base_url.rstrip('/')}/works?{urlencode(params)}"
        res = requests.get(url, timeout=timeout_s)
        if res.status_code != 200:
            raise RuntimeError(f"OpenAlex works fetch failed | status={res.status_code} url={url}")
        data = res.json() or {}
        if not isinstance(data, dict):
            break
        results = data.get("results") or []
        if not isinstance(results, list) or not results:
            break
        for r in results:
            if isinstance(r, dict):
                yield r
        meta = data.get("meta") or {}
        if not isinstance(meta, dict):
            break
        next_cursor = meta.get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor.strip() or next_cursor == cursor:
            break
        cursor = next_cursor


def _pick_venue_display_name(work: Dict[str, Any]) -> Optional[str]:
    primary = work.get("primary_location")
    if isinstance(primary, dict):
        src = primary.get("source")
        if isinstance(src, dict):
            dn = src.get("display_name")
            if isinstance(dn, str) and dn.strip():
                return dn.strip()
    return None


def _pick_work_url(work: Dict[str, Any]) -> Optional[str]:
    ids = work.get("ids")
    if isinstance(ids, dict):
        for k in ("openalex", "doi", "url"):
            v = ids.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    wid = work.get("id")
    if isinstance(wid, str) and wid.strip():
        return wid.strip()
    return None


def _extract_authors_and_institutions(work: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    authors: List[str] = []
    insts: List[str] = []
    authorships = work.get("authorships")
    if isinstance(authorships, list):
        for a in authorships:
            if not isinstance(a, dict):
                continue
            author = a.get("author")
            if isinstance(author, dict):
                dn = author.get("display_name")
                if isinstance(dn, str) and dn.strip():
                    authors.append(dn.strip())
            institutions = a.get("institutions")
            if isinstance(institutions, list):
                for inst in institutions:
                    if not isinstance(inst, dict):
                        continue
                    dn = inst.get("display_name")
                    if isinstance(dn, str) and dn.strip():
                        insts.append(dn.strip())

    def uniq(xs: List[str], *, limit: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in xs:
            k = x.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
            if len(out) >= limit:
                break
        return out

    return uniq(authors, limit=30), uniq(insts, limit=30)


def run_venue_radar(
    *,
    config_path: Path,
    limit: Optional[int] = None,
    no_fetch: bool = False,
    purge: bool = False,
    mark_local: bool = True,
    lookback_days: Optional[int] = None,
) -> int:
    cfg = load_app_config(Path(config_path))

    radar_collection = "openalex_abstracts"
    store = ChromaStore(cfg=cfg, embedder=None, collection_name=radar_collection)
    radar_db = VenueRadarStore(db_path=default_venue_radar_db_path(cfg))

    if purge:
        logger.info("Purging radar stores | chroma_collection=%s sqlite=%s", radar_collection, radar_db.db_path)
        try:
            store.reset_collection()
        except Exception as e:
            logger.warning("Failed to reset radar Chroma collection: %s", e)
        try:
            n = radar_db.purge_all()
            logger.info("Purged venue_abstracts rows: %s", n)
        except Exception as e:
            logger.warning("Failed to purge radar sqlite: %s", e)

    if no_fetch:
        logger.info(
            "no_fetch enabled: initialized stores only | chroma_collection=%s sqlite=%s",
            radar_collection,
            radar_db.db_path,
        )
        return 0

    target_venues = list(cfg.venue_radar.target_venues or [])
    if not target_venues:
        logger.info("No target venues configured (venue_radar.target_venues is empty).")
        return 0

    lb = int(lookback_days) if lookback_days is not None else int(cfg.venue_radar.lookback_days)
    if lb <= 0:
        raise ValueError(f"lookback_days must be positive, got {lb}")
    from_date = (dt.date.today() - dt.timedelta(days=lb)).isoformat()

    embedder = get_embedder(config_path=Path(config_path))
    store.bind_embedder(embedder)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, object]] = []
    radar_openalex_ids: Dict[str, str] = {}
    radar_dois: Dict[str, str] = {}
    radar_titles: Dict[str, str] = {}

    processed = 0
    for venue_id in target_venues:
        logger.info("Fetching OpenAlex works | venue=%s from=%s", venue_id, from_date)
        for work in iter_openalex_works_by_venue(
            venue_source_id=venue_id,
            from_publication_date=from_date,
            mailto=cfg.openalex.mailto,
            api_key=cfg.openalex.api_key,
        ):
            wid = work.get("id")
            if not isinstance(wid, str) or not wid.strip():
                continue
            title = work.get("title")
            title_s = title.strip() if isinstance(title, str) else ""
            abstract = restore_abstract_from_inverted_index(work.get("abstract_inverted_index"))
            if not abstract:
                continue
            pub_date = work.get("publication_date")
            pub_date_s = pub_date.strip() if isinstance(pub_date, str) and pub_date.strip() else None
            venue_name = _pick_venue_display_name(work) or None
            url = _pick_work_url(work) or None

            text = (title_s + "\n\n" + abstract).strip() if title_s else abstract.strip()
            if not text:
                continue

            chunk_id = wid.strip()
            doi = work.get("doi")
            doi_s = doi.strip() if isinstance(doi, str) and doi.strip() else ""

            authors, insts = _extract_authors_and_institutions(work)
            radar_db.upsert_abstract(
                id=chunk_id,
                openalex_id=chunk_id,
                doi=doi_s or None,
                title=title_s,
                abstract=abstract,
                publication_date=pub_date_s,
                venue_name=venue_name,
                url=url,
                authors_json=json.dumps(authors, ensure_ascii=False),
                institutions_json=json.dumps(insts, ensure_ascii=False),
            )

            ids.append(chunk_id)
            docs.append(text)
            radar_openalex_ids[chunk_id] = chunk_id
            radar_dois[chunk_id] = doi_s
            radar_titles[chunk_id] = title_s
            metas.append(
                {
                    "radar_id": chunk_id,
                    "title": title_s,
                    "publication_date": pub_date_s or "",
                    "venue_name": venue_name or "",
                    "url": url or "",
                    "source_id": str(venue_id),
                    "from_date": from_date,
                    "authors": ", ".join(authors[:10]),
                }
            )
            processed += 1
            if limit is not None and processed >= int(limit):
                break
        if limit is not None and processed >= int(limit):
            break

    if not docs:
        logger.info("No eligible works found (with non-empty abstract) in the lookback window.")
        return 0

    logger.info("Embedding + upsert | count=%s collection=%s", len(docs), radar_collection)
    vecs = embedder.embed_texts(docs)
    embs = [list(v) for v in vecs]
    store.upsert_chunks(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    if mark_local:
        marked_at = dt.datetime.now(dt.timezone.utc).isoformat()
        logger.info("Scanning local vault presence for radar works...")
        matches = build_radar_local_match_map(
            cfg=cfg,
            radar_ids=ids,
            radar_openalex_ids=radar_openalex_ids,
            radar_dois=radar_dois,
            radar_titles=radar_titles,
        )
        updated = radar_db.mark_local_scan(radar_ids=ids, matches=matches, marked_at=marked_at)
        logger.info("Local presence marking done. matched=%s updated=%s", len(matches), int(updated))
    return len(docs)

