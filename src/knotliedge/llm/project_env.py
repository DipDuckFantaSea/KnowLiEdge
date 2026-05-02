from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_project_dotenv(project_root: Path, *, filename: str = ".env") -> None:
    """Load ``KEY=value`` pairs from ``{project_root}/{filename}`` into ``os.environ``.

    - Skips empty lines and lines starting with ``#`` (after strip).
    - Does **not** override keys already present in ``os.environ``.
    - Values are stripped; surrounding single/double quotes on values are removed.

    Args:
        project_root: Repository root directory.
        filename: Env file name relative to project root (default ``.env``).
    """

    path = Path(project_root).resolve() / filename
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def get_compatible_chat_api_key() -> Optional[str]:
    """Bearer token for OpenAI-compatible Chat Completions (OpenAI or DashScope).

    Prefers **``DASHSCOPE_API_KEY``** (阿里云百炼 / 兼容模式文档约定)，否则使用 **``OPENAI_API_KEY``**。
    """

    for name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        s = os.environ.get(name)
        if s is None:
            continue
        t = str(s).strip()
        if t:
            return t
    return None


def get_deepseek_chat_api_key() -> Optional[str]:
    """Bearer token for DeepSeek OpenAI-compatible Chat Completions.

    Prefers ``DEEPSEEK_API_KEY``, then ``OPENAI_API_KEY`` (some setups reuse one key slot).
    """

    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        s = os.environ.get(name)
        if s is None:
            continue
        t = str(s).strip()
        if t:
            return t
    return None


def get_openai_api_key() -> Optional[str]:
    """Deprecated name: same as :func:`get_compatible_chat_api_key`."""

    return get_compatible_chat_api_key()


def get_quick_translate_api_key() -> Optional[str]:
    """API key for the *translation-only* qwen call (fast, cheap, isolated quota)."""

    s = os.environ.get("QUICK_TRANSLATE_API_KEY")
    if s is None:
        return None
    t = str(s).strip()
    return t or None
