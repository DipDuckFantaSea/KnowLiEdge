from __future__ import annotations

import copy
import json
import os
import time
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Union

from knotliedge.llm.project_env import get_compatible_chat_api_key, get_deepseek_chat_api_key

# 默认：阿里云百炼 OpenAI 兼容模式（华北北京），见仓库根目录 AliAIPlatfor.md
DEFAULT_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
USER_QUERY_PLACEHOLDER = "__USER_QUERY__"


def _bjt_timestamp() -> str:
    # Beijing time is UTC+8.
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime(time.time() + 8 * 3600))


def _append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(obj), ensure_ascii=False) + "\n")


def _compat_chat_base_url() -> str:
    u = (os.environ.get("OPENAI_BASE_URL") or DEFAULT_COMPAT_BASE_URL).strip().rstrip("/")
    return u or DEFAULT_COMPAT_BASE_URL.rstrip("/")


def _apply_model_override(body: MutableMapping[str, Any]) -> None:
    for env_name in ("OPENAI_MODEL", "DASHSCOPE_MODEL"):
        env_m = os.environ.get(env_name)
        if env_m and str(env_m).strip():
            body["model"] = str(env_m).strip()
            return


def load_chat_completion_template(path: Path) -> Dict[str, Any]:
    """Load a JSON request body for ``POST /v1/chat/completions``."""

    p = Path(path).resolve()
    raw = p.read_text(encoding="utf-8", errors="strict")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"template must be a JSON object: {p}")
    return dict(obj)


def inject_user_query(body: Dict[str, Any], query: str, *, placeholder: str = USER_QUERY_PLACEHOLDER) -> Dict[str, Any]:
    """Return a deep copy of ``body`` with ``placeholder`` replaced by ``query`` in message contents."""

    out = copy.deepcopy(body)
    msgs = out.get("messages")
    if not isinstance(msgs, list):
        raise ValueError("template.messages must be a list")
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str) and placeholder in c:
            m["content"] = c.replace(placeholder, query)
        msgs[i] = m
    out["messages"] = msgs
    return out


def post_chat_completion(
    body: Mapping[str, Any],
    *,
    api_key: str,
    timeout_s: float = 120.0,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """POST ``body`` to OpenAI-compatible ``.../chat/completions``; return parsed JSON object."""

    key = (api_key or "").strip()
    if not key:
        raise ValueError("Chat Completions API key is empty")
    if base_url is None:
        base = _compat_chat_base_url()
    else:
        base = str(base_url).strip().rstrip("/")
    url = f"{base}/chat/completions"
    payload_s = json.dumps(dict(body), ensure_ascii=False)
    payload = payload_s.encode("utf-8")

    feedback_path_s = (os.environ.get("KNOTLIEDGE_CHAT_FEEDBACK_PATH") or "").strip()
    feedback_doc_id = (os.environ.get("KNOTLIEDGE_CHAT_FEEDBACK_DOC_ID") or "").strip()
    feedback_path = Path(feedback_path_s).resolve() if feedback_path_s else None
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    ctx = ssl.create_default_context()
    debug_dir = (os.environ.get("KNOTLIEDGE_CHAT_DEBUG_DIR") or "").strip()
    debug_tag = str(os.environ.get("KNOTLIEDGE_CHAT_DEBUG_TAG") or "").strip()
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    debug_base: Path | None = None
    if debug_dir:
        try:
            debug_base = Path(debug_dir).resolve()
            debug_base.mkdir(parents=True, exist_ok=True)
        except Exception:
            debug_base = None
    if debug_base is not None:
        try:
            safe_body = dict(body)
            # Never write secrets: remove API key and redact any explicit auth header.
            safe_body.pop("api_key", None)
            fname = f"chat_{ts}_{debug_tag}_request.json".replace("__", "_").strip("_")
            (debug_base / fname).write_text(payload_s, encoding="utf-8")
            meta = {
                "ts": ts,
                "url": url,
                "timeout_s": float(timeout_s),
                "payload_bytes": len(payload),
                "note": "request body written separately; API key never logged",
            }
            (debug_base / f"chat_{ts}_{debug_tag}_meta.json".replace("__", "_").strip("_")).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    ts_bjt = _bjt_timestamp()
    if feedback_path is not None:
        try:
            _append_jsonl(
                feedback_path,
                {
                    "ts_bjt": ts_bjt,
                    "doc_id": feedback_doc_id,
                    "kind": "request",
                    "url": url,
                    "timeout_s": float(timeout_s),
                    "payload_bytes": int(len(payload)),
                    "body": payload_s,
                },
            )
        except Exception:
            pass
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s), context=ctx) as resp:
            raw_out = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:4000]
        except Exception:
            pass
        if debug_base is not None:
            try:
                err_obj = {"kind": "HTTPError", "code": int(e.code), "reason": str(e.reason), "detail": detail}
                (debug_base / f"chat_{ts}_{debug_tag}_error.json".replace("__", "_").strip("_")).write_text(
                    json.dumps(err_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        if feedback_path is not None:
            try:
                _append_jsonl(
                    feedback_path,
                    {
                        "ts_bjt": ts_bjt,
                        "doc_id": feedback_doc_id,
                        "kind": "response",
                        "ok": False,
                        "error_kind": "HTTPError",
                        "http_code": int(e.code),
                        "reason": str(e.reason),
                        "detail": detail,
                    },
                )
                _append_jsonl(
                    feedback_path,
                    {"ts_bjt": ts_bjt, "doc_id": feedback_doc_id, "kind": "tokens", "ok": False, "error_kind": "HTTPError"},
                )
            except Exception:
                pass
        raise RuntimeError(f"Chat Completions HTTP {e.code}: {detail or e.reason}") from e
    except urllib.error.URLError as e:
        if debug_base is not None:
            try:
                err_obj = {"kind": "URLError", "error": str(e)}
                (debug_base / f"chat_{ts}_{debug_tag}_error.json".replace("__", "_").strip("_")).write_text(
                    json.dumps(err_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        if feedback_path is not None:
            try:
                _append_jsonl(
                    feedback_path,
                    {"ts_bjt": ts_bjt, "doc_id": feedback_doc_id, "kind": "response", "ok": False, "error_kind": "URLError", "error": str(e)},
                )
                _append_jsonl(
                    feedback_path,
                    {"ts_bjt": ts_bjt, "doc_id": feedback_doc_id, "kind": "tokens", "ok": False, "error_kind": "URLError"},
                )
            except Exception:
                pass
        raise RuntimeError(f"Chat Completions request failed: {e}") from e
    except Exception as e:
        if debug_base is not None:
            try:
                err_obj = {"kind": "Exception", "error": str(e)}
                (debug_base / f"chat_{ts}_{debug_tag}_error.json".replace("__", "_").strip("_")).write_text(
                    json.dumps(err_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        if feedback_path is not None:
            try:
                _append_jsonl(
                    feedback_path,
                    {"ts_bjt": ts_bjt, "doc_id": feedback_doc_id, "kind": "response", "ok": False, "error_kind": "Exception", "error": str(e)},
                )
                _append_jsonl(
                    feedback_path,
                    {"ts_bjt": ts_bjt, "doc_id": feedback_doc_id, "kind": "tokens", "ok": False, "error_kind": "Exception"},
                )
            except Exception:
                pass
        raise

    if feedback_path is not None:
        try:
            _append_jsonl(
                feedback_path,
                {"ts_bjt": ts_bjt, "doc_id": feedback_doc_id, "kind": "response", "ok": True, "raw": raw_out},
            )
        except Exception:
            pass
    if debug_base is not None:
        try:
            (debug_base / f"chat_{ts}_{debug_tag}_response_raw.json".replace("__", "_").strip("_")).write_text(
                raw_out, encoding="utf-8"
            )
        except Exception:
            pass
    try:
        parsed = json.loads(raw_out)
    except json.JSONDecodeError as e:
        if debug_base is not None:
            try:
                err_obj = {"kind": "JSONDecodeError", "error": str(e), "raw_prefix": raw_out[:1000]}
                (debug_base / f"chat_{ts}_{debug_tag}_error.json".replace("__", "_").strip("_")).write_text(
                    json.dumps(err_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        if feedback_path is not None:
            try:
                _append_jsonl(
                    feedback_path,
                    {
                        "ts_bjt": ts_bjt,
                        "doc_id": feedback_doc_id,
                        "kind": "tokens",
                        "ok": False,
                        "error_kind": "JSONDecodeError",
                        "error": str(e),
                        "raw_prefix": raw_out[:1000],
                    },
                )
            except Exception:
                pass
        raise RuntimeError(f"Chat Completions response is not JSON: {raw_out[:500]}") from e
    if not isinstance(parsed, dict):
        raise RuntimeError("Chat Completions response JSON root must be an object")
    if feedback_path is not None:
        try:
            usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
            _append_jsonl(
                feedback_path,
                {
                    "ts_bjt": ts_bjt,
                    "doc_id": feedback_doc_id,
                    "kind": "tokens",
                    "ok": True,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "usage": usage,
                },
            )
        except Exception:
            pass
    return parsed


def extract_assistant_text(response: Mapping[str, Any]) -> str:
    """Return assistant text from ``choices[0].message``.

    Handles OpenAI-style ``content`` strings, multimodal ``content`` part lists, and
    DeepSeek-style ``reasoning_content`` when ``content`` is empty.
    """

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    c0 = choices[0]
    if not isinstance(c0, dict):
        return ""
    msg = c0.get("message")
    if not isinstance(msg, dict):
        return ""

    def _from_content_field(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, list):
            parts: list[str] = []
            for block in raw:
                if isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text") is not None:
                        parts.append(str(block.get("text")))
                    elif isinstance(block.get("content"), str):
                        parts.append(str(block.get("content")))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts).strip()
        return str(raw).strip()

    text = _from_content_field(msg.get("content"))
    if text:
        return text
    rc = msg.get("reasoning_content")
    if rc is not None and str(rc).strip():
        return str(rc).strip()
    return ""


def run_template_chat(
    *,
    project_root: Path,
    template_relative: Union[str, Path] = Path("templates") / "openai_chat" / "local_research_keyword.json",
    query: str,
    timeout_s: float = 120.0,
    api_key: str | None = None,
) -> str:
    """Load template under ``project_root``, inject ``query``, POST, return assistant ``content`` string."""

    root = Path(project_root).resolve()
    tpl = root / Path(template_relative)
    body = load_chat_completion_template(tpl)
    body = inject_user_query(body, query)
    _apply_model_override(body)
    model_id = str(body.get("model") or "").strip().lower()
    if model_id.startswith("deepseek"):
        ds_base = (
            (os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1")
            .strip()
            .rstrip("/")
        )
        key = (api_key or "").strip() or (get_deepseek_chat_api_key() or "").strip()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY is not set (template uses a DeepSeek model)")
        resp = post_chat_completion(body, api_key=key, timeout_s=timeout_s, base_url=ds_base)
    else:
        key = (api_key or "").strip() or (get_compatible_chat_api_key() or "").strip()
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY or OPENAI_API_KEY is not set")
        resp = post_chat_completion(body, api_key=key, timeout_s=timeout_s, base_url=None)
    return extract_assistant_text(resp)
