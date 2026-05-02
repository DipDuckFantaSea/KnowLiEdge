from __future__ import annotations

import hashlib
from pathlib import Path


def compute_doc_id(pdf_path: Path) -> str:
    """Compute a stable doc_id for a PDF file.

    The ID is based on absolute path + file mtime + file size. It is stable
    for unchanged files and changes when a file is modified or replaced.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Hex sha1 string.
    """
    p = pdf_path.resolve()
    stat = p.stat()
    raw = f"{p}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

