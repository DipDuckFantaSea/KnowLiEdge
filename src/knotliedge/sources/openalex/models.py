from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class OpenAlexVenue:
    """Minimal OpenAlex venue representation (offline cache friendly)."""

    id: str
    display_name: Optional[str] = None
    issn_l: Optional[str] = None
    issn: Optional[List[str]] = None
    host_organization: Optional[str] = None


@dataclass(frozen=True)
class OpenAlexWork:
    """Minimal OpenAlex work representation (offline cache friendly)."""

    id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    cited_by_count: Optional[int] = None
    venue: Optional[OpenAlexVenue] = None
    concepts: Optional[List[str]] = None
    raw: Optional[Dict[str, object]] = None

