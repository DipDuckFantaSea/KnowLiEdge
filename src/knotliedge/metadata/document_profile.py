from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from knotliedge.llm.openai_chat import run_template_chat

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def strip_frontmatter(md_text: str) -> str:
    s = md_text or ""
    if not s.startswith("---"):
        return s
    # Find the closing '---' fence.
    lines = s.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return s
    end = None
    for i in range(1, min(len(lines), 4000)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return s
    return "\n".join(lines[end + 1 :]).lstrip("\n")


def extract_toc(md_text: str, *, max_items: int = 120) -> str:
    items: List[str] = []
    for m in _HEADING_RE.finditer(md_text or ""):
        hashes = m.group(1) or "#"
        title = (m.group(2) or "").strip()
        if not title:
            continue
        level = max(1, min(6, len(hashes)))
        if level > 3:
            continue
        indent = "  " * (level - 1)
        items.append(f"{indent}- {title}")
        if len(items) >= int(max_items):
            break
    return "\n".join(items).strip()


_ABSTRACT_TITLES = {
    "abstract",
    "摘要",
}
_CONCLUSION_TITLES = {
    "conclusion",
    "conclusions",
    "结论",
    "summary",
    "讨论",
    "discussion",
}


def _norm_title(t: str) -> str:
    s = (t or "").strip().lower()
    s = re.sub(r"[\s:：\-–—]+", " ", s).strip()
    return s


def _find_section_ranges(md_text: str) -> List[Tuple[int, int, int, str]]:
    """Return list of (start_idx, end_idx, level, title)."""
    matches = list(_HEADING_RE.finditer(md_text or ""))
    out: List[Tuple[int, int, int, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text or "")
        level = len(m.group(1) or "#")
        title = (m.group(2) or "").strip()
        out.append((start, end, level, title))
    return out


def extract_named_sections(md_text: str, *, max_chars_each: int = 2200) -> Tuple[str, str]:
    text = md_text or ""
    ranges = _find_section_ranges(text)
    abstract = ""
    conclusion = ""
    for start, end, _lvl, title in ranges:
        key = _norm_title(title)
        body = (text[start:end] or "").strip()
        if not body:
            continue
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if any(key == _norm_title(x) or key.startswith(_norm_title(x) + " ") for x in _ABSTRACT_TITLES):
            if not abstract:
                abstract = body[: int(max_chars_each)].rstrip()
        if any(key == _norm_title(x) or key.startswith(_norm_title(x) + " ") for x in _CONCLUSION_TITLES):
            if not conclusion:
                conclusion = body[: int(max_chars_each)].rstrip()
        if abstract and conclusion:
            break
    return abstract.strip(), conclusion.strip()


def build_profile_draft(md_text: str, *, max_chars_total: int = 12000) -> str:
    core = strip_frontmatter(md_text)
    toc = extract_toc(core)
    abstract, conclusion = extract_named_sections(core)
    parts: List[str] = []
    if toc:
        parts.append("【目录骨架】\n" + toc)
    if abstract:
        parts.append("【Abstract/摘要】\n" + abstract)
    if conclusion:
        parts.append("【Conclusion/结论】\n" + conclusion)
    draft = "\n\n".join(parts).strip()
    if int(max_chars_total) > 0 and len(draft) > int(max_chars_total):
        draft = draft[: int(max_chars_total)].rstrip() + "…"
    return draft


def compress_profile_with_llm(
    *,
    project_root: Path,
    draft: str,
    timeout_s: float = 120.0,
    max_chars: int = 6000,
    max_retries: int = 4,
    backoff_s: float = 1.5,
) -> Optional[str]:
    d = (draft or "").strip()
    if not d:
        return None
    attempts = max(1, int(max_retries) + 1)
    last_err: Optional[BaseException] = None
    for i in range(attempts):
        try:
            out = run_template_chat(
                project_root=Path(project_root),
                template_relative=Path("templates") / "openai_chat" / "document_profile_compress.json",
                query=d,
                timeout_s=float(timeout_s),
            )
            t = (out or "").strip()
            if not t:
                raise RuntimeError("empty model response content")
            if int(max_chars) > 0 and len(t) > int(max_chars):
                t = t[: int(max_chars)].rstrip() + "…"
            return t
        except Exception as e:
            last_err = e
            if i >= attempts - 1:
                break
            # Exponential backoff with jitter; keep small to avoid long stalls.
            sleep_s = float(backoff_s) * (1.8**i) + random.random() * 0.4
            logger.warning(
                "document_profile LLM compress failed; retrying (%s/%s) after %.2fs | %s",
                i + 1,
                attempts,
                sleep_s,
                e,
            )
            time.sleep(max(0.0, sleep_s))

    logger.warning("document_profile LLM compress failed; fallback to heuristic | %s", last_err)
    return None


@dataclass(frozen=True)
class DocumentProfileResult:
    profile: str
    used_llm: bool
    attempts: int
    fallback_used: bool


def build_document_profile_with_meta(
    *,
    project_root: Path,
    md_text: str,
    timeout_s: float = 120.0,
    draft_max_chars_total: int = 12000,
    compressed_max_chars: int = 6000,
    max_retries: int = 4,
    backoff_s: float = 1.5,
) -> DocumentProfileResult:
    draft = build_profile_draft(md_text, max_chars_total=int(draft_max_chars_total))
    # compress_profile_with_llm returns None on failure after retries.
    compressed = compress_profile_with_llm(
        project_root=Path(project_root),
        draft=draft,
        timeout_s=float(timeout_s),
        max_chars=int(compressed_max_chars),
        max_retries=int(max_retries),
        backoff_s=float(backoff_s),
    )
    if compressed is not None and compressed.strip():
        return DocumentProfileResult(profile=compressed.strip(), used_llm=True, attempts=max(1, int(max_retries) + 1), fallback_used=False)
    return DocumentProfileResult(profile=(draft or "").strip(), used_llm=False, attempts=max(1, int(max_retries) + 1), fallback_used=True)


def build_document_profile(
    *,
    project_root: Path,
    md_text: str,
    timeout_s: float = 120.0,
    draft_max_chars_total: int = 12000,
    compressed_max_chars: int = 6000,
) -> str:
    draft = build_profile_draft(md_text, max_chars_total=int(draft_max_chars_total))
    compressed = compress_profile_with_llm(
        project_root=Path(project_root),
        draft=draft,
        timeout_s=float(timeout_s),
        max_chars=int(compressed_max_chars),
    )
    return (compressed or draft).strip()

