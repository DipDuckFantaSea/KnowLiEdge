from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from knotliedge.config.types import ChunkingConfig


_FRONTMATTER_RE = re.compile(r"^---\s*$", re.MULTILINE)

logger = logging.getLogger(__name__)

# Default reference section titles (extended set).
# Note: this is only used when indexing/chunking; citation graph extraction has its own parser.
DEFAULT_REFERENCE_SECTION_TITLES: List[str] = [
    # English
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


@dataclass(frozen=True)
class MarkdownDoc:
    """A markdown document loaded from vault."""

    doc_id: str
    short_name: str
    source_md: Path
    title: str
    body: str


@dataclass(frozen=True)
class Chunk:
    """A chunk extracted from a Markdown document."""

    chunk_index: int
    chunk_id: str
    doc_id: str
    source_md: Path
    section: Optional[str]
    text: str


def _split_frontmatter(md_text: str) -> Tuple[dict, str]:
    # Expect:
    # ---
    # yaml
    # ---
    # body
    lines = md_text.splitlines()
    if len(lines) >= 3 and lines[0].strip() == "---":
        try:
            end = None
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    end = i
                    break
            if end is None:
                return {}, md_text
            fm_text = "\n".join(lines[1:end])
            fm = yaml.safe_load(fm_text) or {}
            body = "\n".join(lines[end + 1 :]).lstrip("\n")
            if isinstance(fm, dict):
                return fm, body
        except Exception:
            return {}, md_text
    return {}, md_text


def split_frontmatter(markdown: str) -> Tuple[Dict[str, Any], str]:
    """Split leading YAML frontmatter from markdown body (public API).

    Returns:
        (frontmatter_dict, body). If no valid ``---`` / ``---`` block, returns ``({}, markdown)``.
    """

    return _split_frontmatter(markdown)


def load_markdown_doc(path: Path) -> MarkdownDoc:
    """Load markdown + frontmatter."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    fm, body = _split_frontmatter(text)
    doc_id = str(fm.get("doc_id") or "")
    short_name = str(fm.get("short_name") or "")
    title = str(fm.get("title") or path.stem)
    if not doc_id:
        # fallback: still indexable but won't match the contract perfectly
        doc_id = path.stem
    if not short_name:
        short_name = doc_id
    return MarkdownDoc(doc_id=doc_id, short_name=short_name, source_md=path, title=title, body=body)


def _iter_blocks(body: str) -> Iterable[Tuple[Optional[str], str]]:
    """Yield (section, block_text) using heading boundaries."""
    current_section: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            yield current_section, "\n".join(buf).strip()
            buf = []

    for line in body.splitlines():
        if line.startswith("#"):
            # heading starts a new block
            for item in flush():
                yield item
            current_section = line.lstrip("#").strip() or current_section
            buf.append(line)
        else:
            buf.append(line)

    for item in flush():
        yield item


def separate_main_text_and_references(
    full_md_text: str, *, reference_section_titles: Sequence[str]
) -> Tuple[str, str]:
    """Split a markdown document into main text and references section (best-effort).

    This is designed to prevent citation contamination in downstream RAG indexing:
    the references list is often rich in keywords (e.g. "Q-factor") but is not
    the explanatory content we want to retrieve.

    Detection rule:
    - Find the first markdown heading line (levels #, ##, ###) whose title matches
      any of ``reference_section_titles`` (case-insensitive, surrounding spaces ignored).
    - Everything before that heading is ``main_text``.
    - That heading and everything after it is ``references_text``.

    Args:
        full_md_text: Full markdown body (frontmatter already removed).
        reference_section_titles: Allowed section titles that mark the start of references.

    Returns:
        Tuple (main_text, references_text). If no boundary is found, references_text is "".
    """
    text = full_md_text or ""
    titles = [str(t).strip() for t in (reference_section_titles or []) if str(t).strip()]
    if not titles:
        return text.strip(), ""

    # Match heading lines: "# References" or "## 参考文献" etc.
    # We normalize by casefold on the captured title.
    heading_re = re.compile(r"^(#{1,3})\s*(?P<title>[^#\n\r]+?)\s*$", re.MULTILINE)
    title_set = {t.casefold() for t in titles}

    for m in heading_re.finditer(text):
        raw_title = (m.group("title") or "").strip().casefold()
        if raw_title in title_set:
            split_index = int(m.start())
            main_text = text[:split_index].strip()
            references_text = text[split_index:].strip()
            return main_text, references_text

    return text.strip(), ""


def chunk_markdown(doc: MarkdownDoc, cfg: ChunkingConfig) -> List[Chunk]:
    """Chunk markdown doc into overlapping text chunks.

    Strategy (stage-1):
    - Respect heading-based blocks when possible.
    - Within a block, further split by paragraphs if too large.
    - Apply simple char-based overlap to keep continuity.
    """
    target = int(cfg.target_chars)
    overlap = int(cfg.overlap_chars)
    min_len = int(cfg.min_chunk_chars)

    body = doc.body
    if bool(getattr(cfg, "exclude_reference_sections", False)):
        titles = getattr(cfg, "reference_section_titles", None)
        if titles is None:
            titles = DEFAULT_REFERENCE_SECTION_TITLES
        main_text, references_text = separate_main_text_and_references(body, reference_section_titles=titles)
        if references_text:
            logger.info(
                "Excluded references section from chunking: doc_id=%s main_chars=%s ref_chars=%s",
                doc.doc_id,
                len(main_text),
                len(references_text),
            )
        body = main_text

    chunks: List[Chunk] = []
    idx = 0

    def push(section: Optional[str], text: str) -> None:
        nonlocal idx
        t = text.strip()
        if len(t) < min_len:
            return
        chunk_id = f"{doc.doc_id}:{idx}"
        chunks.append(
            Chunk(
                chunk_index=idx,
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source_md=doc.source_md,
                section=section,
                text=t,
            )
        )
        idx += 1

    for section, block in _iter_blocks(body):
        if not block.strip():
            continue
        if len(block) <= target:
            push(section, block)
            continue

        paras = [p.strip() for p in re.split(r"\n{2,}", block) if p.strip()]
        buf = ""
        for p in paras:
            if not buf:
                buf = p
                continue
            if len(buf) + 2 + len(p) <= target:
                buf = f"{buf}\n\n{p}"
            else:
                push(section, buf)
                # start next with overlap tail
                tail = buf[-overlap:] if overlap > 0 else ""
                buf = f"{tail}\n\n{p}".strip()
        if buf:
            push(section, buf)

    return chunks

