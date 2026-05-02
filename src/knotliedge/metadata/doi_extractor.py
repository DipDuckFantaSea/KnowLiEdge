from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from knotliedge.chunking.md_chunker import DEFAULT_REFERENCE_SECTION_TITLES, separate_main_text_and_references

logger = logging.getLogger(__name__)


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\barXiv:(\d{4}\.\d{4,5})\b", re.IGNORECASE)
_PMID_RE = re.compile(r"\bPMID\s*:\s*(\d{4,12})\b", re.IGNORECASE)


def _clean_doi(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^https?://doi\.org/", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip(").,;:]}>\"' \t\r\n")
    return s


def _pick_doi_near_anchors(text: str) -> Optional[str]:
    lines = (text or "").splitlines()
    anchors = ("doi", "doi:", "https://doi.org/", "http://doi.org/")
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(a in low for a in anchors):
            window = "\n".join(lines[max(0, i - 1) : min(len(lines), i + 2)])
            m = _DOI_RE.search(window)
            if m:
                return _clean_doi(m.group(0))
    return None


def _extract_doi_from_pdf_metadata(pdf_path: str) -> Optional[str]:
    p = Path(pdf_path) if pdf_path else None
    if p is None or not str(p).strip() or not p.exists() or not p.is_file():
        return None
    try:
        from pypdf import PdfReader  # local import: optional dependency
    except Exception as e:
        logger.error("pypdf not available for DOI extraction | %s", e)
        return None

    try:
        reader = PdfReader(str(p))
        meta = getattr(reader, "metadata", None)
        if not meta:
            return None
        # metadata may behave like a dict; values can be non-str
        parts = []
        try:
            items = list(getattr(meta, "items")())
        except Exception:
            items = []
        for _k, v in items:
            if v is None:
                continue
            parts.append(str(v))
        blob = "\n".join(parts)
        m = _DOI_RE.search(blob)
        if not m:
            return None
        return _clean_doi(m.group(0))
    except Exception as e:
        logger.warning("Failed to read PDF metadata for DOI | pdf=%s | %s", p, e)
        return None


@dataclass(frozen=True)
class _MdSignals:
    doi: Optional[str]
    arxiv: Optional[str]
    pmid: Optional[str]


def _extract_signals_from_markdown(markdown_text: str) -> _MdSignals:
    # 1) cut references to avoid DOI pollution
    try:
        main_text, _refs = separate_main_text_and_references(
            markdown_text or "", reference_section_titles=DEFAULT_REFERENCE_SECTION_TITLES
        )
    except Exception:
        main_text = markdown_text or ""

    # Fallback split if caller didn't provide titles: use simple heuristic.
    if main_text == (markdown_text or ""):
        low = (markdown_text or "").lower()
        idx = -1
        for key in ("\n# references", "\n## references", "\n### references", "\n# 参考文献", "\n## 参考文献", "\n### 参考文献"):
            j = low.find(key)
            if j != -1 and (idx == -1 or j < idx):
                idx = j
        if idx != -1:
            main_text = (markdown_text or "")[:idx]

    # 2) restrict scanning window: focus on header/abstract zone
    lines = (main_text or "").splitlines()
    head = "\n".join(lines[:200])
    head = head[:20000]

    doi = _pick_doi_near_anchors(head)
    if doi is None:
        m = _DOI_RE.search(head)
        doi = _clean_doi(m.group(0)) if m else None

    arxiv_m = _ARXIV_RE.search(head)
    pmid_m = _PMID_RE.search(head)
    return _MdSignals(
        doi=doi,
        arxiv=arxiv_m.group(1) if arxiv_m else None,
        pmid=pmid_m.group(1) if pmid_m else None,
    )


def _search_openalex_doi_by_title(*, title: str, mailto: str) -> Optional[str]:
    q = (title or "").strip()
    if not q:
        return None
    try:
        import requests
    except Exception as e:
        logger.error("requests not available for OpenAlex title search | %s", e)
        return None

    params = {"search": q, "per-page": 5, "mailto": mailto}
    url = f"https://api.openalex.org/works?{urlencode(params)}"
    try:
        res = requests.get(url, timeout=20)
        if res.status_code != 200:
            logger.warning("OpenAlex title search failed | status=%s url=%s", res.status_code, url)
            return None
        data = res.json() or {}
        results = data.get("results") or []
        for r in results:
            doi = r.get("doi")
            if isinstance(doi, str) and doi.strip():
                return _clean_doi(doi)
        return None
    except Exception as e:
        logger.warning("OpenAlex title search error | %s", e)
        return None


def extract_or_find_doi(pdf_path: str, markdown_text: str, title: str) -> str | None:
    """Extract DOI with a 3-level fallback strategy.

    Level 1: PDF metadata (pypdf).
    Level 2: Markdown regex in high-confidence window (references excluded).
    Level 3: OpenAlex fuzzy match by title (returns DOI if present).

    Args:
        pdf_path: Path to the source PDF (may be empty/None-like).
        markdown_text: Markdown body text (ideally without frontmatter).
        title: Document title for OpenAlex fallback.

    Returns:
        DOI string (without doi.org prefix) if found, else None.
    """
    # Level 1
    doi = _extract_doi_from_pdf_metadata(pdf_path)
    if doi:
        return doi

    # Level 2
    sig = _extract_signals_from_markdown(markdown_text)
    if sig.doi:
        return sig.doi
    if sig.arxiv or sig.pmid:
        logger.info("Found non-DOI id in markdown head | arxiv=%s pmid=%s", sig.arxiv, sig.pmid)

    # Level 3
    mailto = "937246371@qq.com"
    return _search_openalex_doi_by_title(title=title, mailto=mailto)

