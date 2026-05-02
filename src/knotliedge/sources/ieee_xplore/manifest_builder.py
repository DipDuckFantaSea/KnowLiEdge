"""Resolve DOIs to IEEE arnumbers via Crossref and OpenAlex (no IEEE HTML hit)."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

_ARNUMBER_RE = re.compile(
    r"ieeexplore\.ieee\.org/(?:[^/\s\"'<>]+/)*document/(\d+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ManifestEntry:
    """One row for ``ieee_xplore_pdfs download --manifest``."""

    arnumber: str
    doi: str
    title: str
    document_url: str
    stamp_url: str
    pdf_filename: str
    status: str

    def to_jsonl_line(self) -> str:
        """Serialize as a single JSON Lines record."""
        return json.dumps(asdict(self), ensure_ascii=False)


def arnumber_from_text(blob: str) -> Optional[str]:
    """Extract first IEEE Xplore document arnumber from free text or URL.

    Args:
        blob: HTML, URL list, or similar string.

    Returns:
        First matched digit arnumber, or None.
    """
    m = _ARNUMBER_RE.search(blob or "")
    if not m:
        return None
    return str(m.group(1)).strip()


def _clean_doi(doi: str) -> str:
    s = (doi or "").strip()
    for p in ("https://doi.org/", "http://doi.org/"):
        if s.lower().startswith(p):
            s = s[len(p) :].strip()
    return s


def _crossref_headers(mailto: str) -> Dict[str, str]:
    m = (mailto or "").strip()
    ua = f"KnotLiEdge/0.1 (mailto:{m})" if m else "KnotLiEdge/0.1"
    return {"User-Agent": ua}


def _urls_from_crossref_message(msg: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    u = msg.get("URL")
    if isinstance(u, str) and u.strip():
        out.append(u.strip())
    links = msg.get("link")
    if isinstance(links, list):
        for item in links:
            if not isinstance(item, dict):
                continue
            lu = item.get("URL")
            if isinstance(lu, str) and lu.strip():
                out.append(lu.strip())
    return out


def _urls_from_openalex_work(work: Dict[str, Any]) -> List[str]:
    out: List[str] = []

    def add(s: object) -> None:
        if isinstance(s, str) and s.strip():
            out.append(s.strip())

    add(work.get("doi"))
    pl = work.get("primary_location")
    if isinstance(pl, dict):
        add(pl.get("landing_page_url"))
        add(pl.get("pdf_url"))
    bol = work.get("best_oa_location")
    if isinstance(bol, dict):
        add(bol.get("landing_page_url"))
        add(bol.get("pdf_url"))
    oa = work.get("open_access")
    if isinstance(oa, dict):
        add(oa.get("oa_url"))
    for loc in work.get("locations") or []:
        if not isinstance(loc, dict):
            continue
        add(loc.get("landing_page_url"))
        add(loc.get("pdf_url"))
    return out


def _openalex_title(work: Dict[str, Any]) -> str:
    t = work.get("title")
    return str(t).strip() if isinstance(t, str) else ""


def fetch_crossref_urls(doi: str, *, mailto: str, timeout_s: float = 30.0) -> List[str]:
    """GET Crossref work and collect URL fields.

    Args:
        doi: Bare or prefixed DOI.
        mailto: Polite Pool mailto for User-Agent.
        timeout_s: HTTP timeout.

    Returns:
        List of URL strings (may be empty).
    """
    try:
        import requests
    except Exception as e:
        logger.error("requests missing for Crossref | %s", e)
        return []

    doi_s = _clean_doi(doi)
    if not doi_s:
        return []
    enc = quote(doi_s, safe="")
    url = f"https://api.crossref.org/works/{enc}"
    try:
        r = requests.get(url, headers=_crossref_headers(mailto), timeout=timeout_s)
        if r.status_code != 200:
            logger.info("Crossref non-200 | status=%s doi=%s", r.status_code, doi_s)
            return []
        data = r.json() or {}
        msg = data.get("message")
        if not isinstance(msg, dict):
            return []
        return _urls_from_crossref_message(msg)
    except Exception as e:
        logger.warning("Crossref fetch error | doi=%s | %s", doi_s, e)
        return []


def fetch_openalex_work_blob(
    doi: str, *, mailto: str, api_key: Optional[str], timeout_s: float = 35.0
) -> Optional[Dict[str, Any]]:
    """GET OpenAlex work by DOI (full JSON).

    Args:
        doi: Bare or prefixed DOI.
        mailto: Required Polite Pool parameter.
        api_key: Optional OpenAlex API key.
        timeout_s: HTTP timeout.

    Returns:
        Work object dict, or None.
    """
    try:
        import requests
    except Exception as e:
        logger.error("requests missing for OpenAlex | %s", e)
        return None

    doi_s = _clean_doi(doi)
    if not doi_s:
        return None
    from urllib.parse import urlencode

    q = {"mailto": (mailto or "").strip()}
    if api_key and str(api_key).strip():
        q["api_key"] = str(api_key).strip()
    url = f"https://api.openalex.org/works/https://doi.org/{doi_s}?{urlencode(q)}"
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code != 200:
            logger.info("OpenAlex non-200 | status=%s doi=%s", r.status_code, doi_s)
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("OpenAlex fetch error | doi=%s | %s", doi_s, e)
        return None


def build_manifest_entry_for_doi(
    doi: str,
    *,
    mailto: str,
    openalex_api_key: Optional[str],
    between_metadata_sources_sleep_s: float = 1.0,
) -> Optional[ManifestEntry]:
    """Resolve a single DOI to a ``ManifestEntry`` if an IEEE document id is found.

    Crossref is tried first, then OpenAlex. Sleeps ``between_metadata_sources_sleep_s``
    between those two calls (polite to third-party APIs).

    Args:
        doi: DOI string.
        mailto: For Crossref User-Agent and OpenAlex query.
        openalex_api_key: Optional OpenAlex key.
        between_metadata_sources_sleep_s: Pause between Crossref and OpenAlex.

    Returns:
        ``ManifestEntry`` or None if arnumber cannot be resolved.
    """
    doi_s = _clean_doi(doi)
    if not doi_s:
        return None

    title = ""
    blob_parts: List[str] = []

    cr_urls = fetch_crossref_urls(doi_s, mailto=mailto)
    blob_parts.extend(cr_urls)
    time.sleep(max(0.0, float(between_metadata_sources_sleep_s)))

    work = fetch_openalex_work_blob(doi_s, mailto=mailto, api_key=openalex_api_key)
    if isinstance(work, dict):
        title = _openalex_title(work)
        blob_parts.extend(_urls_from_openalex_work(work))

    combined = "\n".join(blob_parts)
    arn = arnumber_from_text(combined)
    if not arn:
        logger.info("no IEEE arnumber for doi=%s", doi_s)
        return None

    doc_url = f"https://ieeexplore.ieee.org/document/{arn}"
    stamp = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arn}"
    pdf_name = f"ieee_{arn}.pdf"
    return ManifestEntry(
        arnumber=arn,
        doi=doi_s,
        title=title,
        document_url=doc_url,
        stamp_url=stamp,
        pdf_filename=pdf_name,
        status="pending",
    )


def build_entries_for_dois(
    dois: Iterable[str],
    *,
    mailto: str,
    openalex_api_key: Optional[str],
    metadata_sleep_s: float,
    between_metadata_sources_sleep_s: float = 1.0,
) -> List[ManifestEntry]:
    """Build manifest rows for many DOIs (sequential, polite).

    Args:
        dois: Iterable of DOI strings.
        mailto: Polite Pool mailto.
        openalex_api_key: Optional OpenAlex API key.
        metadata_sleep_s: Sleep after finishing each DOI (before starting the next).
        between_metadata_sources_sleep_s: Sleep between Crossref and OpenAlex per DOI.

    Returns:
        List of entries with arnumber resolved (skips DOIs without IEEE URL).
    """
    out: List[ManifestEntry] = []
    for raw in dois:
        doi_s = _clean_doi(str(raw))
        if not doi_s:
            continue
        ent = build_manifest_entry_for_doi(
            doi_s,
            mailto=mailto,
            openalex_api_key=openalex_api_key,
            between_metadata_sources_sleep_s=between_metadata_sources_sleep_s,
        )
        if ent:
            out.append(ent)
        time.sleep(max(0.0, float(metadata_sleep_s)))
    return out
