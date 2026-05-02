from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class Frontmatter:
    """Minimal frontmatter for a parsed paper."""

    doc_id: str
    short_name: str
    source_pdf: str
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    parsed_at: str
    parser: str
    version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "short_name": self.short_name,
            "source_pdf": self.source_pdf,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "parsed_at": self.parsed_at,
            "parser": self.parser,
            "version": self.version,
        }


def now_iso8601() -> str:
    """Return current UTC time in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def wrap_with_frontmatter(frontmatter: Frontmatter, markdown_body: str) -> str:
    """Build a Markdown file content with YAML frontmatter.

    Args:
        frontmatter: Frontmatter instance.
        markdown_body: Markdown content (without frontmatter).

    Returns:
        Full Markdown string with YAML frontmatter.
    """
    fm = yaml.safe_dump(frontmatter.to_dict(), sort_keys=False, allow_unicode=True).strip()
    body = (markdown_body or "").lstrip("\n")
    return f"---\n{fm}\n---\n\n{body}\n"


def write_markdown_with_frontmatter(
    *,
    output_path: Path,
    frontmatter: Frontmatter,
    markdown_body: str,
) -> None:
    """Write Markdown with YAML frontmatter to disk.

    Args:
        output_path: Target .md path.
        frontmatter: Frontmatter object.
        markdown_body: Markdown body.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        wrap_with_frontmatter(frontmatter, markdown_body),
        encoding="utf-8",
    )

