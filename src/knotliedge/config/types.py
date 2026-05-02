from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class EnvironmentConfig:
    """Runtime environment identity.

    This eliminates heuristic path branching such as checking ``'sandbox'`` in path parts.
    """

    name: str  # "prod" | "sandbox"

    @property
    def is_sandbox(self) -> bool:
        return str(self.name).strip().lower() == "sandbox"


@dataclass(frozen=True)
class PathsConfig:
    """Paths configuration.

    All paths are resolved against the project root.
    """

    raw_pdf_dir: Path
    markdown_vault_dir: Path
    markdown_assets_dir: Path
    chroma_db_dir: Path


@dataclass(frozen=True)
class MarkdownConfig:
    """Markdown vault configuration."""

    vault_filename_pattern: str
    frontmatter_version: str


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking configuration."""

    target_chars: int
    overlap_chars: int
    min_chunk_chars: int
    exclude_reference_sections: bool
    reference_section_titles: List[str]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    model_name_or_path: str
    device: str
    batch_size: int
    normalize_embeddings: bool


@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB configuration.

    Vectors are accessed only via **HTTP** (``chromadb.HttpClient``) to a standalone
    Chroma server process. ``paths.chroma_db_dir`` is the persistence directory that
    the server must be started with (``chroma run --path ...``), not an embedded DB
    path inside the app process.
    """

    collection_name: str
    http_host: str = "localhost"
    http_port: int = 37651


@dataclass(frozen=True)
class OpenAlexConfig:
    """OpenAlex online API configuration."""

    mailto: str
    api_key: Optional[str]


@dataclass(frozen=True)
class VenueRadarConfig:
    """Parallel Venue Radar configuration (quarantine abstracts)."""

    target_venues: List[str]
    lookback_days: int


@dataclass(frozen=True)
class IeeeXploreConfig:
    """IEEE Xplore stamp/PDF fetch: strict rate limits and session defaults.

    Use only on networks where you are authorized to download (e.g. institutional).
    """

    min_interval_s: float
    user_agent: str
    trust_env_for_proxy: bool
    max_retries: int
    backoff_base_s: float


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    project_root: Path
    environment: EnvironmentConfig
    paths: PathsConfig
    markdown: MarkdownConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    openalex: OpenAlexConfig
    venue_radar: VenueRadarConfig
    ieee_xplore: IeeeXploreConfig

