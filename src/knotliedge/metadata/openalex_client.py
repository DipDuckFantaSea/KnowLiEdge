from __future__ import annotations

import logging
from typing import Dict, List, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


def _clean_doi(doi: str) -> str:
    s = (doi or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :].strip()
    return s


def _extract_journal_name(work: Dict[str, object]) -> Optional[str]:
    host_venue = work.get("host_venue")
    if isinstance(host_venue, dict):
        name = host_venue.get("display_name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    primary_location = work.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            name = source.get("display_name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    locations = work.get("locations")
    if isinstance(locations, list) and locations:
        for loc in locations:
            if not isinstance(loc, dict):
                continue
            source = loc.get("source")
            if isinstance(source, dict):
                name = source.get("display_name")
                if isinstance(name, str) and name.strip():
                    return name.strip()

    return None


def _extract_authors(work: Dict[str, object]) -> List[str]:
    authorships = work.get("authorships")
    out: List[str] = []
    if isinstance(authorships, list):
        for a in authorships:
            if not isinstance(a, dict):
                continue
            author = a.get("author")
            if isinstance(author, dict):
                name = author.get("display_name")
                if isinstance(name, str) and name.strip():
                    out.append(name.strip())
    # de-dup keep order
    seen = set()
    uniq: List[str] = []
    for n in out:
        k = n.casefold()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    return uniq[:50]


def fetch_openalex_metadata(doi: str) -> dict | None:
    """Fetch structured metadata from OpenAlex by DOI.

    Args:
        doi: DOI string, with or without `https://doi.org/` prefix.

    Returns:
        Dict with keys: id, cited_by_count, publication_year, journal_name; or None if not found.
    """
    doi_s = _clean_doi(doi)
    if not doi_s:
        return None

    mailto = "937246371@qq.com"
    base = "https://api.openalex.org"

    try:
        import requests
    except Exception as e:
        logger.error("requests not available for OpenAlex | %s", e)
        return None

    def parse_work(work: Dict[str, object]) -> Optional[Dict[str, object]]:
        oa_id = work.get("id")
        cited_by_count = work.get("cited_by_count")
        publication_year = work.get("publication_year")
        if not isinstance(oa_id, str) or not oa_id.strip():
            return None
        out: Dict[str, object] = {"id": oa_id}
        title = work.get("title")
        if isinstance(title, str) and title.strip():
            out["title"] = title.strip()
        authors = _extract_authors(work)
        if authors:
            out["authors"] = authors
        if isinstance(cited_by_count, int):
            out["cited_by_count"] = cited_by_count
        elif cited_by_count is not None:
            try:
                out["cited_by_count"] = int(cited_by_count)  # type: ignore[arg-type]
            except Exception:
                pass
        if isinstance(publication_year, int):
            out["publication_year"] = publication_year
        elif publication_year is not None:
            try:
                out["publication_year"] = int(publication_year)  # type: ignore[arg-type]
            except Exception:
                pass
        jn = _extract_journal_name(work)
        if jn:
            out["journal_name"] = jn
        return out

    # 1) direct endpoint
    url1 = f"{base}/works/https://doi.org/{doi_s}?{urlencode({'mailto': mailto})}"
    try:
        r1 = requests.get(url1, timeout=25)
        if r1.status_code == 200:
            data = r1.json() or {}
            if isinstance(data, dict):
                parsed = parse_work(data)
                if parsed:
                    return parsed
        else:
            logger.info("OpenAlex direct DOI lookup non-200 | status=%s doi=%s", r1.status_code, doi_s)
    except Exception as e:
        logger.warning("OpenAlex direct DOI lookup error | doi=%s | %s", doi_s, e)

    # 2) filter endpoint
    params = {"filter": f"doi:{doi_s}", "per-page": 1, "mailto": mailto}
    url2 = f"{base}/works?{urlencode(params)}"
    try:
        r2 = requests.get(url2, timeout=25)
        if r2.status_code != 200:
            logger.warning("OpenAlex filter lookup failed | status=%s doi=%s", r2.status_code, doi_s)
            return None
        data2 = r2.json() or {}
        if not isinstance(data2, dict):
            return None
        results = data2.get("results") or []
        if not isinstance(results, list) or not results:
            return None
        top = results[0]
        if not isinstance(top, dict):
            return None
        return parse_work(top)
    except Exception as e:
        logger.warning("OpenAlex filter lookup error | doi=%s | %s", doi_s, e)
        return None

