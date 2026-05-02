from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import List, Optional, Tuple


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _ascii_simplify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).strip()
    return s


def _abbr_words(s: str, *, max_len: int = 12) -> str:
    """Abbreviate a phrase into a compact token."""
    s = _ascii_simplify(s)
    if not s:
        return ""
    words = [w for w in s.split() if w]
    if not words:
        return ""
    # Prefer initialism if multiple words
    if len(words) >= 2:
        abbr = "".join(w[0].upper() for w in words if w)
    else:
        abbr = words[0][: max_len].capitalize()
    return abbr[:max_len]


def _extract_title(md_text: str) -> Optional[str]:
    # Prefer first markdown H1
    for line in md_text.splitlines()[:80]:
        line = line.strip()
        if line.startswith("# "):
            t = line[2:].strip()
            if t:
                return t
    # Fallback: first non-empty line
    for line in md_text.splitlines()[:40]:
        t = line.strip()
        if t:
            return t
    return None


def _extract_year(md_text: str) -> Optional[int]:
    head = "\n".join(md_text.splitlines()[:120])
    m = _YEAR_RE.search(head)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_first_author(md_text: str) -> Optional[str]:
    # Heuristic: look for a line starting with "Authors" or "Author"
    for line in md_text.splitlines()[:120]:
        raw = line.strip()
        low = raw.lower()
        if low.startswith("authors") or low.startswith("author"):
            # split by ":" then by comma
            parts = raw.split(":", 1)
            cand = parts[1] if len(parts) == 2 else raw
            cand = cand.strip()
            if not cand:
                continue
            first = re.split(r"[;,]| and ", cand, maxsplit=1)[0].strip()
            if first:
                # take last token as surname if looks like western name
                toks = _ascii_simplify(first).split()
                if toks:
                    return toks[-1].capitalize()
    return None


def _topic_from_title(title: str) -> str:
    # pick 1-2 salient tokens from title (ascii)
    s = _ascii_simplify(title).lower()
    stop = {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "for",
        "in",
        "on",
        "with",
        "to",
        "via",
        "from",
        "based",
        "using",
        "study",
        "analysis",
        "method",
        "methods",
        "approach",
        "towards",
        "toward",
    }
    toks = [t for t in re.split(r"\s+", s) if t and t not in stop and len(t) >= 4]
    if not toks:
        return "Topic"
    top = toks[:2]
    return "".join(w.capitalize() for w in top)[:16]


@dataclass(frozen=True)
class ShortNameParts:
    author: str
    year: str
    venue_abbr: str
    topic: str
    suffix: str


def generate_short_name(
    *,
    markdown_text: str,
    venue: Optional[str],
    pdf_path: Path,
) -> Tuple[str, ShortNameParts]:
    """Generate a unique, human-readable short name for a paper.

    Format:
      {Author}{Year}{VenueAbbr}{Topic}-{suffix}

    The suffix is a short hash for uniqueness.

    Args:
        markdown_text: Markdown content (body or full text).
        venue: Optional venue/journal name if available.
        pdf_path: Original PDF path (used for fallback uniqueness).

    Returns:
        (short_name, parts)
    """
    title = _extract_title(markdown_text) or pdf_path.stem
    author = _extract_first_author(markdown_text) or "Anon"
    year_i = _extract_year(markdown_text)
    year = str(year_i) if year_i else "n.d."
    venue_abbr = _abbr_words(venue or "Doc", max_len=10) or "Doc"
    topic = _topic_from_title(title)

    base = f"{author}{year}{venue_abbr}{topic}"
    base = re.sub(r"[^A-Za-z0-9.]+", "", _ascii_simplify(base)) or "Doc"
    raw = f"{pdf_path.resolve()}|{title}|{author}|{year}|{venue_abbr}|{topic}".encode("utf-8")
    suffix = sha1(raw).hexdigest()[:6]

    short_name = f"{base}-{suffix}"
    return short_name, ShortNameParts(author=author, year=year, venue_abbr=venue_abbr, topic=topic, suffix=suffix)

