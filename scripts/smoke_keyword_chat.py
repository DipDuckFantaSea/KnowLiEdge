"""Smoke test: load repo-root ``.env``, call DeepSeek keyword expand (no secrets printed)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    out = sys.stdout
    if hasattr(out, "reconfigure"):
        try:
            out.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from knotliedge.llm.project_env import get_deepseek_chat_api_key, load_project_dotenv
    from knotliedge.llm.query_keyword_expand import try_expand_query_to_keywords

    load_project_dotenv(root)
    if not get_deepseek_chat_api_key():
        print("FAIL: set DEEPSEEK_API_KEY or OPENAI_API_KEY in repo root .env")
        return 1

    # English query reduces format drift vs. the EN-terms template (strict one-line tokens).
    query = "GaN power device thermal resistance flip-chip package heat spreading"
    kws = try_expand_query_to_keywords(project_root=root, query=query, timeout_s=120.0)
    if not kws:
        print("FAIL: keyword expand returned None (see WARNING logs above)")
        return 2

    print("OK: DeepSeek Chat Completions keyword expand succeeded")
    print("keywords:", " ".join(kws))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
