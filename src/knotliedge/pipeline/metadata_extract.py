from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class PaperMetadata:
    """Best-effort metadata extracted from Markdown text.

    This module intentionally uses lightweight heuristics (no network, no heavy models).
    """

    title: Optional[str]
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def _strip_md(line: str) -> str:
    s = (line or "").strip()
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _head_lines(md_text: str, n: int = 120) -> List[str]:
    return [ln.rstrip("\n") for ln in (md_text or "").splitlines()[: int(n)]]


def _extract_title(lines: List[str]) -> Optional[str]:
    for ln in lines[:80]:
        s = ln.strip()
        if s.startswith("# "):
            t = _strip_md(s)
            if t and len(t) >= 4:
                return t
    for ln in lines[:40]:
        t = _strip_md(ln)
        if t and len(t) >= 4:
            low = t.lower()
            if low in {"abstract", "keywords"}:
                continue
            return t
    return None


def _split_authors(raw: str) -> List[str]:
    s = (raw or "").strip()
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\[[^\]]*\]|\([^\)]*\)|\*+", " ", s)  # footnotes/affiliations markers
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s:
        return []
    parts = re.split(r"\s*(?:,|;| and | & )\s*", s)
    out: List[str] = []
    for p in parts:
        p = p.strip().strip(".")
        if not p:
            continue
        if len(p) <= 2:
            continue
        out.append(p)
    # de-dup keep order
    seen = set()
    uniq: List[str] = []
    for a in out:
        k = a.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(a)
    return uniq[:20]


def _extract_authors(lines: List[str]) -> Tuple[List[str], Optional[str]]:
    # 1) explicit "Authors:" / "Author:" line
    for ln in lines[:160]:
        raw = ln.strip()
        low = raw.lower()
        if low.startswith("authors:") or low.startswith("author:") or low.startswith("authors ") or low.startswith("author "):
            _, _, tail = raw.partition(":")
            cand = (tail or raw).strip()
            authors = _split_authors(cand)
            if authors:
                return authors, cand
    # 2) lines after title until blank/abstract; take the first non-trivial line that looks like names
    title = _extract_title(lines)
    if not title:
        return [], None
    title_idx = None
    for i, ln in enumerate(lines[:80]):
        if _strip_md(ln) == title:
            title_idx = i
            break
    if title_idx is None:
        return [], None
    for ln in lines[title_idx + 1 : title_idx + 10]:
        s = _strip_md(ln)
        if not s:
            break
        low = s.lower()
        if low.startswith("abstract") or low.startswith("keywords"):
            break
        # heuristic: multiple capitalized tokens / separators
        if len(s) >= 6 and ("," in s or " and " in low or "&" in s):
            authors = _split_authors(s)
            if authors:
                return authors, s
    return [], None


def _extract_year(lines: List[str]) -> Optional[int]:
    head = "\n".join(lines[:200])
    m = _YEAR_RE.search(head)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_venue(lines: List[str]) -> Optional[str]:
    head = "\n".join(lines[:220])
    low = head.lower()
    if "arxiv" in low:
        # common: "arXiv:xxxx.xxxxx" or "arXiv preprint"
        for ln in lines[:220]:
            if "arxiv" in ln.lower():
                s = _strip_md(ln)
                return s[:120] if s else "arXiv"
        return "arXiv"
    if _DOI_RE.search(head):
        # Keep venue unknown; DOI alone isn't venue.
        pass
    for ln in lines[:220]:
        s = _strip_md(ln)
        if not s:
            continue
        l = s.lower()
        if l.startswith("proceedings of "):
            return s[:120]
        if l.startswith("journal of ") or "transactions on" in l:
            return s[:120]
        if l.startswith("conference on ") or l.startswith("international conference on "):
            return s[:120]
    return None


def extract_paper_metadata(markdown_text: str) -> PaperMetadata:
    """Extract best-effort title/authors/year/venue from Markdown.

    Args:
        markdown_text: Markdown content.

    Returns:
        PaperMetadata with missing fields as None/[].
    """
    lines = _head_lines(markdown_text, n=240)
    title = _extract_title(lines)
    authors, _authors_raw = _extract_authors(lines)
    year = _extract_year(lines)
    venue = _extract_venue(lines)
    return PaperMetadata(title=title, authors=authors, year=year, venue=venue)

