from __future__ import annotations

from pathlib import Path
from typing import Optional

from knotliedge.llm.openai_chat import run_template_chat
from knotliedge.llm.project_env import get_quick_translate_api_key


def translate_query_to_english(
    *,
    project_root: Path,
    query: str,
    timeout_s: float = 60.0,
    template_relative: str | Path = Path("templates") / "openai_chat" / "quick_translate_en.json",
) -> Optional[str]:
    """Translate a query into English for retrieval; return ``None`` on failure."""

    key = get_quick_translate_api_key()
    if not key:
        return None
    raw = run_template_chat(
        project_root=project_root,
        template_relative=template_relative,
        query=query,
        timeout_s=timeout_s,
        api_key=key,
    )
    out = str(raw or "").strip()
    if not out:
        return None
    # Keep only the first line to avoid accidental drift.
    if "\n" in out or "\r" in out:
        out = out.splitlines()[0].strip()
    return out or None

