from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from openai import OpenAI

from knotliedge.llm.openai_chat import extract_assistant_text
from knotliedge.llm.project_env import load_project_dotenv

logger = logging.getLogger(__name__)


USER_QUERY_PLACEHOLDER = "__USER_QUERY__"


@dataclass(frozen=True)
class LlmConnection:
    """Connection settings for an OpenAI-compatible chat API."""

    base_url: str
    api_key_env: str


def _read_json(path: Path) -> dict[str, Any]:
    p = Path(path).resolve()
    raw = p.read_text(encoding="utf-8", errors="strict")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {p}")
    return dict(obj)


def _deep_replace_placeholder(obj: Any, *, placeholder: str, value: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(placeholder, value) if placeholder in obj else obj
    if isinstance(obj, list):
        return [_deep_replace_placeholder(x, placeholder=placeholder, value=value) for x in obj]
    if isinstance(obj, dict):
        return {k: _deep_replace_placeholder(v, placeholder=placeholder, value=value) for k, v in obj.items()}
    return obj


def load_llm_tasks_config(config_path: Path) -> dict[str, Any]:
    """Load task defaults from a single JSON file."""

    cfg = _read_json(config_path)
    if not isinstance(cfg.get("tasks"), dict):
        raise ValueError("config.tasks must be an object")
    return cfg


def build_request_for_task(
    cfg: Mapping[str, Any],
    *,
    task_id: str,
    user_query: str,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[LlmConnection, dict[str, Any]]:
    """Build (connection, request) for a task and injected user query.

    Args:
        cfg: Loaded config dict.
        task_id: Task identifier in cfg["tasks"].
        user_query: Text injected into the request.
        overrides: Optional shallow overrides merged into the request object.

    Returns:
        (connection, request) where request is the kwargs dict for OpenAI SDK.
    """

    tasks = cfg.get("tasks")
    if not isinstance(tasks, dict) or task_id not in tasks:
        raise KeyError(f"Unknown task_id: {task_id}")
    task = tasks[task_id]
    if not isinstance(task, dict):
        raise ValueError(f"task must be an object: {task_id}")

    placeholder = str(cfg.get("placeholder") or USER_QUERY_PLACEHOLDER)

    conn_obj = task.get("connection")
    if not isinstance(conn_obj, dict):
        raise ValueError(f"task.connection must be an object: {task_id}")
    base_url = str(conn_obj.get("base_url") or "").strip().rstrip("/")
    api_key_env = str(conn_obj.get("api_key_env") or "").strip()
    if not base_url or not api_key_env:
        raise ValueError(f"task.connection.base_url/api_key_env is required: {task_id}")
    conn = LlmConnection(base_url=base_url, api_key_env=api_key_env)

    req_obj = task.get("request")
    if not isinstance(req_obj, dict):
        raise ValueError(f"task.request must be an object: {task_id}")
    req: dict[str, Any] = copy.deepcopy(req_obj)
    req = _deep_replace_placeholder(req, placeholder=placeholder, value=str(user_query))
    if overrides:
        # Shallow merge only; keep templates stable and explicit.
        for k, v in overrides.items():
            req[k] = v
    return conn, req


def run_task(
    *,
    config_path: Path,
    task_id: str,
    user_query: str,
    timeout_s: float = 120.0,
    overrides: Mapping[str, Any] | None = None,
) -> str:
    """Execute a configured task and return assistant message content."""

    # Allow users to put API keys in repo-root `.env` for local runs.
    # We never override existing `os.environ` keys.
    try:
        config_p = Path(config_path).resolve()
        project_root = config_p.parent.parent if config_p.parent.name.lower() == "config" else config_p.parent
        load_project_dotenv(project_root)
    except Exception:
        pass

    cfg = load_llm_tasks_config(config_path)
    conn, req = build_request_for_task(cfg, task_id=task_id, user_query=user_query, overrides=overrides)

    api_key = (os.environ.get(conn.api_key_env) or "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {conn.api_key_env}")

    client = OpenAI(api_key=api_key, base_url=conn.base_url)

    model = req.pop("model", None)
    messages = req.pop("messages", None)
    if not model or not messages:
        raise ValueError("request must include model and messages")

    # Pass through any OpenAI-compatible kwargs, including:
    # - reasoning_effort
    # - extra_body (for DeepSeek thinking)
    # - temperature/max_tokens/stream, etc.
    resp = client.chat.completions.create(model=str(model), messages=messages, timeout=timeout_s, **req)

    try:
        return extract_assistant_text(resp.model_dump())
    except Exception:
        try:
            return str(resp.choices[0].message.content or "")
        except Exception:
            return ""

