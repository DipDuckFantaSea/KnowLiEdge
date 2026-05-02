"""Smoke test: load repo-root ``.env``, call DeepSeek document_profile compress (no secrets printed)."""

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
    from knotliedge.metadata.document_profile import compress_profile_with_llm

    load_project_dotenv(root)
    if not get_deepseek_chat_api_key():
        print("FAIL: set DEEPSEEK_API_KEY or OPENAI_API_KEY in repo root .env")
        return 1

    draft = (
        "【目录骨架】\n"
        "  - Introduction\n"
        "  - Modeling\n"
        "【Abstract/摘要】\n"
        "GaN-on-Si HEMT compact modeling with thermal coupling; benchmark vs TCAD.\n"
    )
    prof = compress_profile_with_llm(
        project_root=root,
        draft=draft,
        timeout_s=120.0,
        max_retries=1,
    )
    if not prof or not str(prof).strip():
        print("FAIL: document_profile compress returned empty (see WARNING logs above)")
        return 2

    print("OK: DeepSeek document_profile compress succeeded")
    preview = str(prof).strip().replace("\r\n", "\n")
    if len(preview) > 600:
        preview = preview[:600] + "…"
    print("profile_preview:\n", preview, sep="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
