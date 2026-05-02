"""Fetch IEEE stamp HTML, resolve iframe PDF URL, stream download."""

from __future__ import annotations

import logging
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from knotliedge.sources.ieee_xplore.rate_limit import HostRateLimiter

logger = logging.getLogger(__name__)

_STAMP_URL_FMT = "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnumber}"
_PDF_MAGIC = b"%PDF"


def stamp_url_for_arnumber(arnumber: str) -> str:
    """Build the IEEE stamp.jsp URL for a document arnumber.

    Args:
        arnumber: Numeric IEEE document id.

    Returns:
        Absolute stamp URL.
    """
    s = str(arnumber).strip()
    if not s.isdigit():
        raise ValueError(f"arnumber must be digits-only, got {arnumber!r}")
    return _STAMP_URL_FMT.format(arnumber=s)


class _IframeSrcParser(HTMLParser):
    """Collect ``iframe[src]`` values from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "iframe":
            return
        low = {str(k).lower(): (v or "") for k, v in attrs}
        src = str(low.get("src", "")).strip()
        if src:
            self.srcs.append(src)


def _looks_like_pdf_href(url: str) -> bool:
    u = url.lower()
    if "pdf" in u:
        return True
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def extract_pdf_url_from_stamp_html(html: str, *, base_url: str) -> Optional[str]:
    """Parse stamp page HTML and return the embedded PDF URL if found.

    Args:
        html: Raw HTML of ``stamp.jsp``.
        base_url: URL used to resolve relative ``iframe[src]`` (usually the stamp URL).

    Returns:
        First iframe ``src`` that looks like a PDF endpoint, else first iframe src, else None.
    """
    parser = _IframeSrcParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as e:
        logger.warning("HTML parse failed for stamp page | %s", e)
        return None
    for raw in parser.srcs:
        joined = urljoin(base_url, raw.strip())
        if _looks_like_pdf_href(joined):
            return joined
    if parser.srcs:
        return urljoin(base_url, parser.srcs[0].strip())
    # Fallback: bare PDF URL in page (last resort)
    m = re.search(r"https?://[^\s\"'<>]+\.pdf(?:\?[^\s\"'<>]*)?", html, flags=re.I)
    if m:
        return m.group(0).strip()
    return None


def merge_mozilla_cookie_file(session: object, cookie_path: Path) -> None:
    """Load Netscape/Mozilla ``cookies.txt`` into a ``requests.Session``.

    Args:
        session: ``requests.Session`` instance.
        cookie_path: Path to Mozilla-format cookie jar file.

    Raises:
        OSError: If cookie file cannot be read.
        ValueError: If ``requests`` helpers are unavailable.
    """
    from http.cookiejar import MozillaCookieJar

    try:
        from requests.utils import dict_from_cookiejar
    except Exception as e:
        raise ValueError("requests is required for cookie merge") from e

    jar = MozillaCookieJar(str(cookie_path))
    jar.load(ignore_discard=True, ignore_expires=True)
    merged = dict_from_cookiejar(jar)
    getattr(session, "cookies").update(merged)


def _request_with_retries(
    session: object,
    method: str,
    url: str,
    *,
    stream: bool = False,
    max_retries: int,
    backoff_base_s: float,
    rate_limiter: Optional[HostRateLimiter],
) -> object:
    """Perform HTTP request with backoff on 429/503 and connection errors.

    Args:
        session: ``requests.Session``.
        method: ``get`` or ``head``.
        url: Target URL.
        stream: Passed to ``requests``.
        max_retries: Max attempts including the first.
        backoff_base_s: Base sleep; attempt n sleeps ``backoff_base_s * 2**n``.
        rate_limiter: If set, ``wait_before_url`` is invoked before each attempt.

    Returns:
        ``requests.Response``.

    Raises:
        RuntimeError: On exhausted retries.
    """
    import requests

    last_exc: Optional[BaseException] = None
    for attempt in range(max(1, int(max_retries))):
        if rate_limiter is not None:
            rate_limiter.wait_before_url(url)
        try:
            fn = getattr(session, method.lower())
            r = fn(url, timeout=120, stream=stream, allow_redirects=True)
            if r.status_code in (429, 503):
                raise requests.HTTPError(f"status={r.status_code}")
            return r
        except BaseException as e:
            last_exc = e
            if attempt >= int(max_retries) - 1:
                break
            sleep_s = float(backoff_base_s) * (2**attempt)
            logger.warning("HTTP retry | url=%s attempt=%s sleep=%.1fs | %s", url, attempt + 1, sleep_s, e)
            time.sleep(sleep_s)
    raise RuntimeError(f"HTTP failed after retries | url={url} | {last_exc!r}") from last_exc


def fetch_stamp_html(
    session: object,
    arnumber: str,
    *,
    max_retries: int,
    backoff_base_s: float,
    rate_limiter: Optional[HostRateLimiter],
) -> tuple[str, str]:
    """GET stamp.jsp and return ``(stamp_url, html_text)``.

    Args:
        session: ``requests.Session`` with User-Agent / cookies if needed.
        arnumber: IEEE document id.
        max_retries: Passed to ``_request_with_retries``.
        backoff_base_s: Passed to ``_request_with_retries``.
        rate_limiter: IEEE spacing before the GET.

    Returns:
        Tuple of stamp URL and decoded HTML body.

    Raises:
        ValueError: On bad arnumber.
        RuntimeError: On HTTP failure.
    """
    stamp_url = stamp_url_for_arnumber(arnumber)
    r = _request_with_retries(
        session,
        "get",
        stamp_url,
        stream=False,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        rate_limiter=rate_limiter,
    )
    status = getattr(r, "status_code", None)
    if status != 200:
        raise RuntimeError(f"stamp GET non-200 | status={status} url={stamp_url}")
    text = getattr(r, "text", "") or ""
    return stamp_url, text


def _is_pdf_response(body_head: bytes, content_type: str) -> bool:
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return True
    return body_head.startswith(_PDF_MAGIC)


def download_pdf_via_stamp(
    session: object,
    arnumber: str,
    out_path: Path,
    *,
    max_retries: int,
    backoff_base_s: float,
    rate_limiter: Optional[HostRateLimiter],
    skip_if_exists: bool = True,
) -> Path:
    """Resolve PDF via stamp iframe and stream bytes to ``out_path``.

    Args:
        session: ``requests.Session``.
        arnumber: IEEE document id.
        out_path: Destination ``.pdf`` path (parent dirs created if missing).
        max_retries: HTTP retry budget per step.
        backoff_base_s: Exponential backoff base.
        rate_limiter: Throttle for IEEE hosts (stamp + typical PDF hosts).
        skip_if_exists: If True and file exists with PDF magic, skip network.

    Returns:
        Resolved ``out_path``.

    Raises:
        RuntimeError: If PDF URL missing, non-PDF body, or I/O errors after logging.
    """
    out_path = out_path.resolve()
    if skip_if_exists and out_path.is_file():
        try:
            head = out_path.read_bytes()[:5]
            if head.startswith(_PDF_MAGIC):
                logger.info("skip existing PDF | %s", out_path)
                return out_path
        except OSError as e:
            logger.warning("could not read existing file | %s | %s", out_path, e)

    stamp_url, html = fetch_stamp_html(
        session,
        arnumber,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        rate_limiter=rate_limiter,
    )
    pdf_url = extract_pdf_url_from_stamp_html(html, base_url=stamp_url)
    if not pdf_url:
        raise RuntimeError(f"no PDF iframe/url found | arnumber={arnumber}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    def _download_to_tmp() -> None:
        r = _request_with_retries(
            session,
            "get",
            pdf_url,
            stream=True,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            rate_limiter=rate_limiter,
        )
        if getattr(r, "status_code", 0) != 200:
            raise RuntimeError(f"pdf GET non-200 | status={getattr(r, 'status_code', None)} url={pdf_url}")
        ct = ""
        try:
            ct = str(getattr(r, "headers", {}).get("Content-Type", "") or "")
        except Exception:
            ct = ""
        first = b""
        try:
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    if len(first) < 5:
                        need = 5 - len(first)
                        first += chunk[:need]
                    f.write(chunk)
        finally:
            try:
                getattr(r, "close", lambda: None)()
            except Exception:
                pass
        if not _is_pdf_response(first, ct):
            try:
                snippet = first[:200].decode("utf-8", errors="replace")
            except Exception:
                snippet = repr(first)
            raise RuntimeError(
                f"response is not PDF | arnumber={arnumber} content-type={ct!r} head={snippet!r}"
            )

    try:
        _download_to_tmp()
        tmp.replace(out_path)
    except OSError as e:
        logger.error("pdf write failed | out=%s | %s", out_path, e)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise

    logger.info("saved PDF | arnumber=%s | %s", arnumber, out_path)
    return out_path
