from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional


def now_iso8601() -> str:
    """Return current UTC time in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ChunkMetadata:
    """Minimal metadata schema stored in ChromaDB for each chunk."""

    doc_id: str
    short_name: str
    chunk_id: str
    source_md: str
    source_md_mtime_ns: Optional[int]
    section: Optional[str]
    chunk_index: int
    text_len: int
    created_at: str
    doc_hash: Optional[str]

    def to_chroma(self) -> Dict[str, object]:
        """Convert to a dict suitable for Chroma metadata."""
        out: Dict[str, object] = {
            "doc_id": self.doc_id,
            "short_name": self.short_name,
            "chunk_id": self.chunk_id,
            "source_md": self.source_md,
            "source_md_mtime_ns": int(self.source_md_mtime_ns) if self.source_md_mtime_ns is not None else None,
            "section": self.section,
            "chunk_index": int(self.chunk_index),
            "text_len": int(self.text_len),
            "created_at": self.created_at,
        }
        if self.doc_hash is not None:
            out["doc_hash"] = str(self.doc_hash)
        return out

