from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from knotliedge.config.types import AppConfig
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.sources.openalex.models import OpenAlexVenue, OpenAlexWork

logger = setup_logging()


def default_openalex_cache_dir(cfg: AppConfig) -> Path:
    return (cfg.project_root / "data" / "06_openalex_cache").resolve()


class OpenAlexCache:
    """Offline cache for OpenAlex works/venues (no network)."""

    def __init__(self, *, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / "works").mkdir(parents=True, exist_ok=True)
        (self._cache_dir / "venues").mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def works_jsonl_path(self, name: str = "works.jsonl") -> Path:
        return (self._cache_dir / "works" / name).resolve()

    def venues_jsonl_path(self, name: str = "venues.jsonl") -> Path:
        return (self._cache_dir / "venues" / name).resolve()

    def append_works_jsonl(self, works: Iterable[OpenAlexWork], *, name: str = "works.jsonl") -> Path:
        path = self.works_jsonl_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for w in works:
                f.write(json.dumps(asdict(w), ensure_ascii=False) + "\n")
        return path

    def iter_works_jsonl(self, *, name: str = "works.jsonl") -> Iterator[OpenAlexWork]:
        path = self.works_jsonl_path(name)
        if not path.exists():
            return iter(())
        def _iter() -> Iterator[OpenAlexWork]:
            for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = (ln or "").strip()
                if not s:
                    continue
                try:
                    raw = json.loads(s)
                    yield OpenAlexWork(**raw)
                except Exception:
                    continue
        return _iter()

