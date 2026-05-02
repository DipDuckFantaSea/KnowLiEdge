from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchPollConfig:
    """Polling strategy for Batch File jobs.

    Args:
        initial_interval_s: First poll interval.
        max_interval_s: Max interval after exponential backoff.
        backoff_factor: Multiply interval when no progress is observed.
        max_polls: Safety cap to avoid endless loops (None = no cap).
    """

    initial_interval_s: float = 10.0
    max_interval_s: float = 60.0
    backoff_factor: float = 1.7
    max_polls: Optional[int] = None


@dataclass(frozen=True)
class BatchRunResult:
    batch_id: str
    status: str
    input_file_id: str
    output_file_id: Optional[str]
    error_file_id: Optional[str]
    raw_batch: Dict[str, Any]


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        try:
            return dict(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return dict(obj.to_dict())
        except Exception:
            pass
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:
        return {"repr": repr(obj)}


def upload_batch_input_file(*, client: OpenAI, jsonl_path: Path) -> str:
    """Upload a JSONL file for Batch File execution (purpose=batch)."""

    p = Path(jsonl_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    file_obj = client.files.create(file=p, purpose="batch")
    d = _as_dict(file_obj)
    fid = str(d.get("id") or "").strip()
    if not fid:
        raise RuntimeError("files.create returned empty id")
    return fid


def create_batch_job(
    *,
    client: OpenAI,
    input_file_id: str,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
    metadata: Optional[Mapping[str, str]] = None,
) -> BatchRunResult:
    """Create a Batch File job."""

    batch = client.batches.create(
        input_file_id=str(input_file_id),
        endpoint=str(endpoint),
        completion_window=str(completion_window),
        metadata=dict(metadata or {}),
    )
    d = _as_dict(batch)
    bid = str(d.get("id") or "").strip()
    if not bid:
        raise RuntimeError("batches.create returned empty id")
    return BatchRunResult(
        batch_id=bid,
        status=str(d.get("status") or ""),
        input_file_id=str(d.get("input_file_id") or input_file_id),
        output_file_id=str(d.get("output_file_id") or "").strip() or None,
        error_file_id=str(d.get("error_file_id") or "").strip() or None,
        raw_batch=d,
    )


def retrieve_batch(*, client: OpenAI, batch_id: str) -> BatchRunResult:
    batch = client.batches.retrieve(str(batch_id))
    d = _as_dict(batch)
    return BatchRunResult(
        batch_id=str(d.get("id") or batch_id),
        status=str(d.get("status") or ""),
        input_file_id=str(d.get("input_file_id") or ""),
        output_file_id=str(d.get("output_file_id") or "").strip() or None,
        error_file_id=str(d.get("error_file_id") or "").strip() or None,
        raw_batch=d,
    )


def wait_for_batch(
    *,
    client: OpenAI,
    batch_id: str,
    poll: BatchPollConfig = BatchPollConfig(),
    on_poll: Optional[callable] = None,
) -> BatchRunResult:
    """Poll a Batch job until it reaches a terminal state.

    The server-side execution is async. We limit polling frequency by exponential backoff.
    """

    interval = max(1.0, float(poll.initial_interval_s))
    last_status = ""
    polls = 0

    while True:
        if poll.max_polls is not None and polls >= int(poll.max_polls):
            raise TimeoutError(f"batch polling exceeded max_polls={poll.max_polls}")
        polls += 1

        r = retrieve_batch(client=client, batch_id=batch_id)
        status = (r.status or "").strip().lower()
        if on_poll is not None:
            try:
                on_poll(r, interval)
            except Exception:
                pass

        if status in {"completed", "failed", "cancelled", "expired"}:
            return r

        if status and status != last_status:
            # Status progressed; reset interval to be responsive.
            interval = max(1.0, float(poll.initial_interval_s))
        else:
            interval = min(float(poll.max_interval_s), interval * float(poll.backoff_factor))
        last_status = status
        time.sleep(max(0.0, interval))


def download_file_text(*, client: OpenAI, file_id: str) -> str:
    """Download a text file content by file_id.

    Batch output/error files are JSONL and returned as UTF-8 text.
    """

    fid = str(file_id or "").strip()
    if not fid:
        return ""
    content = client.files.content(fid)
    # openai-python returns either bytes-like response or wrapper with .read()
    if hasattr(content, "read"):
        raw = content.read()
    else:
        raw = content  # type: ignore[assignment]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def iter_jsonl_lines(text: str) -> Iterator[Dict[str, Any]]:
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def parse_batch_output_jsonl(output_text: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse output JSONL into {custom_id -> response_obj}, plus raw rows list."""

    by_id: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for obj in iter_jsonl_lines(output_text):
        rows.append(obj)
        cid = str(obj.get("custom_id") or "").strip()
        resp = obj.get("response")
        if cid and isinstance(resp, dict):
            by_id[cid] = resp
    return by_id, rows


def parse_batch_error_jsonl(error_text: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse error JSONL into {custom_id -> error_obj}, plus raw rows list."""

    by_id: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for obj in iter_jsonl_lines(error_text):
        rows.append(obj)
        cid = str(obj.get("custom_id") or "").strip()
        err = obj.get("error")
        if cid and isinstance(err, dict):
            by_id[cid] = err
    return by_id, rows


def build_batch_jsonl_rows(
    *,
    custom_ids_and_bodies: Iterable[Tuple[str, Mapping[str, Any]]],
    url: str = "/v1/chat/completions",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cid, body in custom_ids_and_bodies:
        c = str(cid).strip()
        if not c:
            continue
        rows.append({"custom_id": c, "method": "POST", "url": str(url), "body": dict(body)})
    return rows


def write_jsonl(*, path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")

