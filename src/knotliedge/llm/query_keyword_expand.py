from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from knotliedge.llm.openai_chat import run_template_chat

logger = logging.getLogger(__name__)

# One line of space-separated tokens; conservative length.
_MAX_LINE_CHARS = 2000
_MAX_TOKENS = 64
_MAX_TOKEN_LEN = 120

_TOKEN_RE = re.compile(r"^[\w.+\-/:@]+$", re.UNICODE)

_LEADING_JUNK_FIRST = frozenset(
    {
        "we",
        "so",
        "the",
        "here",
        "output",
        "extract",
        "this",
        "let",
        "i",
        "need",
        "your",
        "ok",
        "sure",
        "thus",
        "therefore",
    }
)


def _normalize_line(s: str) -> str:
    t = (s or "").replace("\ufeff", "").strip()
    if "\n" in t or "\r" in t:
        t = t.splitlines()[0].strip()
    # strip common markdown fences / bullets (model drift)
    t = t.strip("`").strip()
    if t.startswith("- ") or t.startswith("* "):
        t = t[2:].strip()
    return t.strip()


def validate_keyword_line(line: str) -> Optional[str]:
    """Return normalized line if valid, else ``None``."""

    line = _normalize_line(line)
    if not line:
        return None
    if len(line) > _MAX_LINE_CHARS:
        return None
    parts = [p for p in line.split() if p]
    if not parts or len(parts) > _MAX_TOKENS:
        return None
    for p in parts:
        if len(p) > _MAX_TOKEN_LEN:
            return None
        if not _TOKEN_RE.match(p):
            return None
    return " ".join(parts)


def _leading_token_junk(parts: List[str]) -> bool:
    if not parts:
        return True
    first = parts[0].lower().strip("`\"'")
    first = first.rstrip(":，,")
    return first in _LEADING_JUNK_FIRST


def pick_valid_keyword_line(raw: str) -> Optional[str]:
    """Pick the best strict keyword line from possibly noisy LLM output.

    Some models (e.g. DeepSeek flash) prepend reasoning prose or quote the user query;
    we try several slices (full text, each line, tails after ``:``, double-quoted spans)
    and return the longest valid token line.
    """

    text = (raw or "").replace("\r\n", "\n").strip()
    if not text:
        return None

    candidates: List[str] = []
    candidates.append(text)
    if ":" in text:
        tail_all = text.split(":")[-1].strip()
        if tail_all:
            candidates.append(tail_all)
    for line in text.split("\n"):
        s = line.strip()
        if s:
            candidates.append(s)
        if ":" in s:
            tail = s.split(":", 1)[-1].strip()
            if tail:
                candidates.append(tail)
    quoted: List[str] = []
    for m in re.finditer(r'"([^"\n]{3,2000})"', text):
        quoted.append(m.group(1).strip())
    for chunk in re.split(r"\.\s+", text):
        c = chunk.strip()
        if c:
            candidates.append(c)

    def _score_line(ok: str) -> int:
        return len(ok)

    def _take_best(pool: List[str]) -> Optional[str]:
        seen_local: set[str] = set()
        best_local: Optional[str] = None
        for c in pool:
            if not c or c in seen_local:
                continue
            seen_local.add(c)
            ok = validate_keyword_line(c)
            if ok is None:
                continue
            parts = ok.split()
            if _leading_token_junk(parts):
                continue
            if best_local is None or _score_line(ok) > _score_line(best_local):
                best_local = ok
        return best_local

    best_q = _take_best(quoted)
    if best_q is not None:
        return best_q
    return _take_best(candidates)


def expand_query_to_keyword_line(
    *,
    project_root: Path,
    query: str,
    timeout_s: float = 120.0,
) -> str:
    """Call compatible Chat Completions to extract English technical terms; return validated one-line keywords.

    Raises:
        RuntimeError: On HTTP / API errors (caller may catch and fall back).
    """

    raw = run_template_chat(
        project_root=project_root,
        template_relative=Path("templates") / "openai_chat" / "local_research_keyword_en_terms.json",
        query=query,
        timeout_s=timeout_s,
    )
    ok = pick_valid_keyword_line(raw)
    if ok is None:
        raise RuntimeError(f"model output failed validation: {raw[:200]!r}")
    return ok


def try_expand_query_to_keywords(
    *,
    project_root: Path,
    query: str,
    timeout_s: float = 120.0,
) -> Optional[List[str]]:
    """Return list of keyword tokens, or ``None`` if expansion/validation fails."""

    try:
        line = expand_query_to_keyword_line(project_root=project_root, query=query, timeout_s=timeout_s)
    except Exception as e:
        logger.warning("LLM keyword expand failed, falling back | %s", e)
        return None
    parts = [p for p in line.split() if p]
    return parts if parts else None
