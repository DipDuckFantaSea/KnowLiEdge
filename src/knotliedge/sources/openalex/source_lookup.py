from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


def normalize_openalex_source_id(raw: object) -> Optional[str]:
    """Normalize OpenAlex Source IDs to canonical URL form.

    Accepts:
    - ``https://openalex.org/S...``
    - ``S...``

    Args:
        raw: Raw ID from config or API payload.

    Returns:
        Canonical Source ID URL, or None if invalid/empty.
    """

    if raw is None:
        return None
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip()
    if not s:
        return None
    if s.startswith("https://openalex.org/S"):
        return s
    if s.startswith("S") and len(s) > 1:
        return f"https://openalex.org/{s}"
    return None


def resolve_openalex_source_id_by_name(
    *,
    name: str,
    mailto: str,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openalex.org",
    timeout_s: int = 25,
) -> Optional[str]:
    """Resolve an OpenAlex Source ID by a venue name (journal / conference).

    This uses the OpenAlex ``/sources`` search endpoint and returns a best-effort match.
    Always include ``mailto`` to enter the Polite Pool.

    Args:
        name: Venue display name, e.g. ``IEEE Transactions on Microwave Theory and Techniques``.
        mailto: Email used for OpenAlex Polite Pool.
        api_key: Optional OpenAlex API key (if available).
        base_url: OpenAlex API base URL.
        timeout_s: Request timeout seconds.

    Returns:
        Canonical Source ID URL (e.g. ``https://openalex.org/S1983995261``) if found,
        else None.
    """

    q = (name or "").strip()
    if not q:
        return None
    mail = (mailto or "").strip()
    if not mail:
        raise ValueError("mailto is required for OpenAlex requests")

    try:
        import requests
    except Exception as e:
        logger.error("requests not available for OpenAlex source lookup | %s", e)
        return None

    params = {
        "search": q,
        "per_page": 10,
        "mailto": mail,
        # reduce payload size
        "select": "id,display_name,type,works_count,cited_by_count",
    }
    k = (api_key or "").strip()
    if k:
        params["api_key"] = k

    url = f"{base_url.rstrip('/')}/sources?{urlencode(params)}"
    try:
        res = requests.get(url, timeout=timeout_s)
        if res.status_code != 200:
            logger.warning("OpenAlex sources search failed | status=%s url=%s", res.status_code, url)
            return None
        data = res.json() or {}
    except Exception as e:
        logger.warning("OpenAlex sources search error | url=%s | %s", url, e)
        return None

    if not isinstance(data, dict):
        return None
    results = data.get("results") or []
    if not isinstance(results, list) or not results:
        return None

    def _pick_id(item: Dict[str, Any]) -> Optional[str]:
        sid = item.get("id")
        return normalize_openalex_source_id(sid)

    q_fold = q.casefold()
    for it in results:
        if not isinstance(it, dict):
            continue
        dn = it.get("display_name")
        if isinstance(dn, str) and dn.strip().casefold() == q_fold:
            sid = _pick_id(it)
            if sid:
                return sid

    top = results[0]
    if isinstance(top, dict):
        return _pick_id(top)
    return None

