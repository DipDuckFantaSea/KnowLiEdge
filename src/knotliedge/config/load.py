from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from knotliedge.config.types import (
    AppConfig,
    ChromaConfig,
    ChunkingConfig,
    EmbeddingConfig,
    EnvironmentConfig,
    IeeeXploreConfig,
    MarkdownConfig,
    OpenAlexConfig,
    PathsConfig,
    VenueRadarConfig,
)
from knotliedge.sources.openalex.source_lookup import normalize_openalex_source_id


def _require_mapping(d: Any, *, name: str) -> Dict[str, Any]:
    if not isinstance(d, dict):
        raise ValueError(f"{name} must be a mapping, got {type(d).__name__}")
    return d


def _optional_str(v: object) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    s = v.strip()
    return s or None


def resolve_project_root_for_config(config_path: Path) -> Path:
    """Find repository root near ``config_path`` for stable relative ``paths:`` resolution.

    Cursor/MCP and other launchers often set ``cwd`` to the user profile or a temp
    directory. Walking upward from the config file directory and stopping at the
    first directory that contains ``pyproject.toml`` yields the KnotLiEdge repo root.

    Args:
        config_path: Path to a YAML config file (absolute or relative).

    Returns:
        Resolved project root directory, or ``Path.cwd()`` if no marker was found.
    """
    start = config_path.resolve().parent
    current = start
    for _ in range(16):
        marker = current / "pyproject.toml"
        if marker.is_file():
            return current.resolve()
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path.cwd().resolve()


def load_app_config(config_path: Path, *, project_root: Optional[Path] = None) -> AppConfig:
    """Load application config from YAML and resolve paths.

    Args:
        config_path: Path to YAML config file.
        project_root: Project root directory. If None, inferred from ``config_path``
            (walk parents for ``pyproject.toml``), else ``Path.cwd()``.

    Returns:
        Parsed AppConfig with resolved absolute paths.
    """
    config_path = config_path.resolve()
    if project_root is None:
        project_root = resolve_project_root_for_config(config_path)
    else:
        project_root = project_root.resolve()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw = _require_mapping(raw, name="config")

    paths_raw = _require_mapping(raw.get("paths", {}), name="paths")
    markdown_raw = _require_mapping(raw.get("markdown", {}), name="markdown")
    chunking_raw = _require_mapping(raw.get("chunking", {}), name="chunking")
    embedding_raw = _require_mapping(raw.get("embedding", {}), name="embedding")
    chroma_raw = _require_mapping(raw.get("chroma", {}), name="chroma")
    openalex_raw = _require_mapping(raw.get("openalex", {}), name="openalex")
    venue_radar_raw = _require_mapping(raw.get("venue_radar", {}), name="venue_radar")
    ieee_raw = raw.get("ieee_xplore", {}) or {}
    if not isinstance(ieee_raw, dict):
        raise ValueError(f"ieee_xplore must be a mapping, got {type(ieee_raw).__name__}")

    default_ref_titles: List[str] = [
        # English (extended)
        "references",
        "reference",
        "bibliography",
        "works cited",
        "literature",
        "literature cited",
        # Chinese
        "参考文献",
        "引用文献",
        "参考资料",
    ]

    paths = PathsConfig(
        raw_pdf_dir=(project_root / Path(paths_raw["raw_pdf_dir"])).resolve(),
        markdown_vault_dir=(project_root / Path(paths_raw["markdown_vault_dir"])).resolve(),
        markdown_assets_dir=(project_root / Path(paths_raw["markdown_assets_dir"])).resolve(),
        chroma_db_dir=(project_root / Path(paths_raw["chroma_db_dir"])).resolve(),
    )
    markdown = MarkdownConfig(
        vault_filename_pattern=str(markdown_raw.get("vault_filename_pattern", "{doc_id}.md")),
        frontmatter_version=str(markdown_raw.get("frontmatter_version", "v1")),
    )
    chunking = ChunkingConfig(
        target_chars=int(chunking_raw.get("target_chars", 1000)),
        overlap_chars=int(chunking_raw.get("overlap_chars", 150)),
        min_chunk_chars=int(chunking_raw.get("min_chunk_chars", 200)),
        exclude_reference_sections=bool(chunking_raw.get("exclude_reference_sections", True)),
        reference_section_titles=[
            str(x).strip()
            for x in (chunking_raw.get("reference_section_titles", default_ref_titles) or [])
            if str(x).strip()
        ],
    )
    embedding = EmbeddingConfig(
        model_name_or_path=str(embedding_raw.get("model_name_or_path", "models/bge-m3")),
        device=str(embedding_raw.get("device", "cuda")),
        batch_size=int(embedding_raw.get("batch_size", 16)),
        normalize_embeddings=bool(embedding_raw.get("normalize_embeddings", True)),
    )
    chroma = ChromaConfig(
        collection_name=str(chroma_raw.get("collection_name", "knotliedge_v1")),
        http_host=str(chroma_raw.get("http_host", "localhost")).strip() or "localhost",
        http_port=int(chroma_raw.get("http_port", 37651)),
    )

    mailto = _optional_str(openalex_raw.get("mailto"))
    if mailto is None:
        raise ValueError("openalex.mailto is required (Polite Pool)")
    openalex = OpenAlexConfig(
        mailto=mailto,
        api_key=_optional_str(openalex_raw.get("api_key")),
    )

    target_raw = venue_radar_raw.get("target_venues", []) or []
    if not isinstance(target_raw, list):
        raise ValueError(f"venue_radar.target_venues must be a list, got {type(target_raw).__name__}")
    norm_targets: List[str] = []
    seen = set()
    for x in target_raw:
        sid = normalize_openalex_source_id(x)
        if sid is None:
            continue
        k = sid.casefold()
        if k in seen:
            continue
        seen.add(k)
        norm_targets.append(sid)

    venue_radar = VenueRadarConfig(
        target_venues=norm_targets,
        lookback_days=int(venue_radar_raw.get("lookback_days", 300)),
    )

    ieee_xplore = IeeeXploreConfig(
        min_interval_s=float(ieee_raw.get("min_interval_s", 60.0)),
        user_agent=str(
            ieee_raw.get(
                "user_agent",
                "KnotLiEdge/0.1 (+https://github.com) ieee-xplore-fetch",
            )
        ).strip()
        or "KnotLiEdge/0.1 ieee-xplore-fetch",
        trust_env_for_proxy=bool(ieee_raw.get("trust_env_for_proxy", False)),
        max_retries=int(ieee_raw.get("max_retries", 4)),
        backoff_base_s=float(ieee_raw.get("backoff_base_s", 5.0)),
    )

    env_name = "sandbox" if ("sandbox" in config_path.parts or "sandbox" in paths.markdown_vault_dir.parts) else "prod"
    environment = EnvironmentConfig(name=env_name)

    return AppConfig(
        project_root=project_root,
        environment=environment,
        paths=paths,
        markdown=markdown,
        chunking=chunking,
        embedding=embedding,
        chroma=chroma,
        openalex=openalex,
        venue_radar=venue_radar,
        ieee_xplore=ieee_xplore,
    )


def config_fingerprint(cfg: AppConfig) -> str:
    """Create a stable short fingerprint for a config.

    Args:
        cfg: AppConfig.

    Returns:
        Short hex digest.
    """
    data = asdict(cfg)
    # Convert Paths to strings
    data["project_root"] = str(cfg.project_root)
    data["paths"] = {k: str(v) for k, v in asdict(cfg.paths).items()}
    b = yaml.safe_dump(data, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()[:12]

