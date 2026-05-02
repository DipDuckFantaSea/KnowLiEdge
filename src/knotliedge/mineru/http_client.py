from __future__ import annotations

import io
import json
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class MinerUTaskSubmitted:
    task_id: str


def _http_bytes(
    *,
    method: str,
    url: str,
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: int = 30,
) -> Tuple[bytes, Dict[str, str]]:
    req = Request(url=url, data=body, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            data = resp.read() or b""
            resp_headers = {str(k).lower(): str(v) for k, v in dict(resp.headers).items()}
            return data, resp_headers
    except HTTPError as e:
        raw = e.read() if hasattr(e, "read") else b""
        msg = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"MinerU HTTP error: {e.code} {e.reason} | {url} | {msg}") from e
    except URLError as e:
        raise RuntimeError(f"MinerU HTTP connection error: {url} | {e}") from e


def _encode_multipart_formdata(
    *,
    files: List[Tuple[str, Path]],
    fields: Dict[str, object],
) -> Tuple[bytes, str]:
    boundary = f"----KnotLiEdgeBoundary{uuid.uuid4().hex}"
    crlf = "\r\n"

    body: List[bytes] = []

    def add_text(name: str, value: str) -> None:
        body.append(f"--{boundary}{crlf}".encode("utf-8"))
        body.append(f'Content-Disposition: form-data; name="{name}"{crlf}{crlf}'.encode("utf-8"))
        body.append(value.encode("utf-8"))
        body.append(crlf.encode("utf-8"))

    def add_file(name: str, path: Path) -> None:
        filename = path.name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body.append(f"--{boundary}{crlf}".encode("utf-8"))
        body.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"{crlf}'.encode("utf-8")
        )
        body.append(f"Content-Type: {content_type}{crlf}{crlf}".encode("utf-8"))
        body.append(path.read_bytes())
        body.append(crlf.encode("utf-8"))

    for k, v in fields.items():
        if isinstance(v, bool):
            add_text(k, "true" if v else "false")
        elif isinstance(v, (int, float, str)):
            add_text(k, str(v))
        elif v is None:
            continue
        else:
            # list/dict -> json
            add_text(k, json.dumps(v, ensure_ascii=False))

    for name, path in files:
        add_file(name, path)

    body.append(f"--{boundary}--{crlf}".encode("utf-8"))
    content_type = f"multipart/form-data; boundary={boundary}"
    return b"".join(body), content_type


def _http_json(
    *,
    method: str,
    url: str,
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: int = 30,
) -> Any:
    try:
        data, _hdrs = _http_bytes(method=method, url=url, body=body, headers=headers, timeout_s=timeout_s)
        if not data:
            return {}
        return json.loads(data.decode("utf-8", errors="replace"))
    except Exception as e:
        # Keep backward-compatible error strings from _http_bytes
        raise e


def health(*, api_url: str, timeout_s: int = 3) -> Dict[str, Any]:
    return _http_json(method="GET", url=f"{api_url.rstrip('/')}/health", timeout_s=timeout_s)


def submit_task(
    *,
    api_url: str,
    pdf_paths: List[Path],
    backend: str = "hybrid-auto-engine",
    parse_method: str = "auto",
    formula_enable: bool = True,
    table_enable: bool = True,
    lang_list: Optional[List[str]] = None,
    return_md: bool = True,
    return_images: bool = False,
    response_format_zip: bool = False,
    timeout_s: int = 30,
) -> MinerUTaskSubmitted:
    # Prefer requests for robust multipart handling; fall back to urllib if unavailable.
    try:
        import requests  # type: ignore

        data: Dict[str, str] = {
            "backend": str(backend),
            "parse_method": str(parse_method),
            "formula_enable": "true" if formula_enable else "false",
            "table_enable": "true" if table_enable else "false",
            "return_md": "true" if return_md else "false",
            "return_images": "true" if return_images else "false",
            "response_format_zip": "true" if response_format_zip else "false",
        }
        # lang_list expects array; send multiple fields with same name.
        files_payload = [("files", (p.name, p.open("rb"), "application/pdf")) for p in pdf_paths]
        for lang in (lang_list or ["ch"]):
            data.setdefault("lang_list", str(lang))
        resp = requests.post(f"{api_url.rstrip('/')}/tasks", data=data, files=files_payload, timeout=timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(f"MinerU HTTP error: {resp.status_code} | {resp.text}")
        res = resp.json() if resp.content else {}
    except Exception:
        files = [("files", p) for p in pdf_paths]
        fields: Dict[str, object] = {
            "backend": backend,
            "parse_method": parse_method,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "lang_list": lang_list or ["ch"],
            "return_md": return_md,
            "return_images": return_images,
            "response_format_zip": response_format_zip,
        }
        body, content_type = _encode_multipart_formdata(files=files, fields=fields)
        # Uploading PDFs can take time; keep a generous timeout.
        res = _http_json(
            method="POST",
            url=f"{api_url.rstrip('/')}/tasks",
            body=body,
            headers={"Content-Type": content_type},
            timeout_s=timeout_s,
        )
    task_id = str(res.get("task_id") or res.get("id") or "")
    if not task_id:
        raise RuntimeError(f"Unexpected MinerU /tasks response: {res}")
    return MinerUTaskSubmitted(task_id=task_id)


def get_task_status(*, api_url: str, task_id: str, timeout_s: int = 10) -> Dict[str, Any]:
    return _http_json(method="GET", url=f"{api_url.rstrip('/')}/tasks/{task_id}", timeout_s=timeout_s)


def get_task_result(*, api_url: str, task_id: str, timeout_s: int = 60) -> Dict[str, Any]:
    return _http_json(method="GET", url=f"{api_url.rstrip('/')}/tasks/{task_id}/result", timeout_s=timeout_s)


def get_task_result_bytes(*, api_url: str, task_id: str, timeout_s: int = 300) -> Tuple[bytes, Dict[str, str]]:
    """Fetch task result as raw bytes (used when response_format_zip=true)."""
    return _http_bytes(method="GET", url=f"{api_url.rstrip('/')}/tasks/{task_id}/result", timeout_s=timeout_s)


def parse_zip_result(*, zip_bytes: bytes) -> Tuple[Optional[str], bytes, Dict[str, bytes]]:
    """Parse MinerU ZIP result bytes.

    Args:
        zip_bytes: Raw ZIP bytes returned by MinerU when response_format_zip=true.

    Returns:
        A tuple (markdown_filename, markdown_bytes, images) where images maps relative paths to bytes.
        If no markdown file exists, markdown_bytes will be b"".
    """
    import zipfile

    md_name: Optional[str] = None
    md_bytes: bytes = b""
    images: Dict[str, bytes] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for info in zf.infolist():
            name = str(info.filename).replace("\\", "/")
            if info.is_dir():
                continue
            lower = name.lower()
            data = zf.read(info)
            if lower.endswith(".md") and md_name is None:
                md_name = name
                md_bytes = data
                continue
            if "/images/" in lower or lower.startswith("images/"):
                images[name] = data

    return md_name, md_bytes, images


def file_parse_sync(
    *,
    api_url: str,
    pdf_path: Path,
    backend: str = "hybrid-auto-engine",
    parse_method: str = "auto",
    formula_enable: bool = True,
    table_enable: bool = True,
    lang_list: Optional[List[str]] = None,
    return_md: bool = True,
    timeout_s: int = 600,
) -> Dict[str, Any]:
    body, content_type = _encode_multipart_formdata(
        files=[("files", pdf_path)],
        fields={
            "backend": backend,
            "parse_method": parse_method,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "lang_list": lang_list or ["ch"],
            "return_md": return_md,
        },
    )
    return _http_json(
        method="POST",
        url=f"{api_url.rstrip('/')}/file_parse",
        body=body,
        headers={"Content-Type": content_type},
        timeout_s=timeout_s,
    )

