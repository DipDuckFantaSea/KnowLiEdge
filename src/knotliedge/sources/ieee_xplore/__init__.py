"""IEEE Xplore stamp page + PDF helpers (institutional access only; strict rate limits)."""

from knotliedge.sources.ieee_xplore.manifest_builder import (
    ManifestEntry,
    arnumber_from_text,
    build_entries_for_dois,
)
from knotliedge.sources.ieee_xplore.rate_limit import HostRateLimiter, is_ieee_rate_limited_host
from knotliedge.sources.ieee_xplore.stamp_pdf import (
    download_pdf_via_stamp,
    extract_pdf_url_from_stamp_html,
    fetch_stamp_html,
    stamp_url_for_arnumber,
)

__all__ = [
    "ManifestEntry",
    "arnumber_from_text",
    "build_entries_for_dois",
    "HostRateLimiter",
    "is_ieee_rate_limited_host",
    "download_pdf_via_stamp",
    "extract_pdf_url_from_stamp_html",
    "fetch_stamp_html",
    "stamp_url_for_arnumber",
]
