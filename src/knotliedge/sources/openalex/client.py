from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from knotliedge.sources.openalex.models import OpenAlexWork


@dataclass(frozen=True)
class OpenAlexClientConfig:
    """Configuration for a future online OpenAlex client."""

    base_url: str = "https://api.openalex.org"
    user_agent: Optional[str] = None
    mailto: Optional[str] = None


class OpenAlexClient:
    """OpenAlex client stub.

    This project can run in environments without external network access.
    Therefore, the online fetching methods are intentionally unimplemented.
    """

    def __init__(self, *, cfg: Optional[OpenAlexClientConfig] = None) -> None:
        self._cfg = cfg or OpenAlexClientConfig()

    @property
    def config(self) -> OpenAlexClientConfig:
        return self._cfg

    def iter_works_by_venue(self, venue_id: str, *, per_page: int = 200) -> Iterator[OpenAlexWork]:
        """Iterate works for a venue (NOT IMPLEMENTED in offline mode)."""
        raise NotImplementedError("Online OpenAlex fetching is not enabled in this environment.")

