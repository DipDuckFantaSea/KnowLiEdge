from __future__ import annotations

from pathlib import Path

from knotliedge.config.types import AppConfig


def ensure_dirs(cfg: AppConfig) -> None:
    """Ensure all configured directories exist.

    Args:
        cfg: AppConfig.

    Returns:
        None.
    """
    cfg.paths.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.markdown_vault_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.markdown_assets_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.chroma_db_dir.mkdir(parents=True, exist_ok=True)

