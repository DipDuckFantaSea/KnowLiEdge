from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExtractedRef:
    ref_id: str
    ref_type: str  # doi / arxiv / url / text
    ref_text: str
    confidence: float


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\b(arxiv:)?(\d{4}\.\d{4,5})(v\d+)?\b", re.IGNORECASE)
_URL_RE = re.compile(r"\bhttps?://[^\s\)>\]]+\b", re.IGNORECASE)

# Keep reference-section boundary titles aligned with the indexing splitter to avoid drift.
# Note: we intentionally keep this local to citation_graph to avoid import cycles.
DEFAULT_REFERENCE_SECTION_TITLES: List[str] = [
    # English (extended)
    "references",
    "reference",
    "bibliography",
    "works cited",
    "literature",
    "literature cited",
    # Chinese
    "参考文献",
    "引用文献",
    "参考资料",
]


def _norm_doi(doi: str) -> str:
    d = (doi or "").strip().lower()
    d = d.removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:")
    return d.strip()


def _norm_arxiv(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = s.removeprefix("arxiv:").strip()
    return s


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def _find_reference_section_lines(md_text: str, *, reference_section_titles: Sequence[str]) -> List[str]:
    """Find candidate reference section lines (best-effort)."""
    lines = (md_text or "").splitlines()
    start = None
    title_set = {str(t).strip().lower() for t in (reference_section_titles or []) if str(t).strip()}
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        if s.startswith("#"):
            t = s.lstrip("#").strip()
            if t in title_set:
                start = i + 1
                break
    if start is None:
        return []
    out: List[str] = []
    for ln in lines[start:]:
        if ln.strip().startswith("#"):
            break
        out.append(ln)
    return out


def _split_reference_entries(lines: Sequence[str]) -> List[str]:
    buf: List[str] = []
    entries: List[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            ent = " ".join([b.strip() for b in buf if b.strip()]).strip()
            if ent:
                entries.append(ent)
            buf = []

    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        is_new = bool(re.match(r"^\s*\[\s*\d+\s*\]\s+|^\s*\d+\.\s+|^\s*\d+\)\s+", s))
        if is_new:
            flush()
        buf.append(s)
    flush()
    return entries


def extract_references_from_markdown(md_text: str) -> List[ExtractedRef]:
    """Extract best-effort references from markdown text.

    Strategy:
    - Prefer hard identifiers: DOI / arXiv / URL.
    - Fall back to hashing raw reference text.

    Args:
        md_text: Markdown content.

    Returns:
        List of ExtractedRef (deduped).
    """
    refs: List[ExtractedRef] = []
    section_lines = _find_reference_section_lines(md_text, reference_section_titles=DEFAULT_REFERENCE_SECTION_TITLES)
    entries = _split_reference_entries(section_lines) if section_lines else []

    # If no explicit reference section, still harvest DOI/URL/arXiv across full text (lower confidence).
    if not entries:
        entries = [md_text]
        base_conf = 0.4
    else:
        base_conf = 0.7

    for ent in entries:
        text = (ent or "").strip()
        if not text:
            continue

        dois = [_norm_doi(m.group(0)) for m in _DOI_RE.finditer(text)]
        urls = [m.group(0).strip() for m in _URL_RE.finditer(text)]
        arxivs = [_norm_arxiv(m.group(0)) for m in _ARXIV_RE.finditer(text)]

        if dois:
            for d in dois:
                if not d:
                    continue
                refs.append(
                    ExtractedRef(
                        ref_id=f"doi:{d}",
                        ref_type="doi",
                        ref_text=text,
                        confidence=min(1.0, base_conf + 0.25),
                    )
                )
            continue
        if arxivs:
            for a in arxivs:
                if not a:
                    continue
                refs.append(
                    ExtractedRef(
                        ref_id=f"arxiv:{a}",
                        ref_type="arxiv",
                        ref_text=text,
                        confidence=min(1.0, base_conf + 0.2),
                    )
                )
            continue
        if urls:
            for u in urls:
                refs.append(
                    ExtractedRef(
                        ref_id=f"url:{u}",
                        ref_type="url",
                        ref_text=text,
                        confidence=min(1.0, base_conf + 0.15),
                    )
                )
            continue

        # Fallback: hash text
        refs.append(
            ExtractedRef(
                ref_id=f"text:{_hash_text(text)}",
                ref_type="text",
                ref_text=text[:1000],
                confidence=base_conf,
            )
        )

    # de-dup keep best confidence
    best: dict[str, ExtractedRef] = {}
    for r in refs:
        cur = best.get(r.ref_id)
        if cur is None or r.confidence > cur.confidence:
            best[r.ref_id] = r
    return list(best.values())

