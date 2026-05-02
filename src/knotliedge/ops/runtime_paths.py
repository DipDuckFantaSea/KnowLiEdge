from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knotliedge.config.types import AppConfig


@dataclass(frozen=True)
class RuntimePaths:
    """Centralized runtime/work directory layout under ``.knotliedge/``."""

    runtime_root: Path
    work_root: Path
    logs_root: Path
    mineru_work_dir: Path
    mineru_logs_dir: Path


def get_runtime_paths(cfg: AppConfig) -> RuntimePaths:
    """Compute standardized runtime paths from config."""
    root = Path(cfg.project_root).resolve() / ".knotliedge"
    work = root / "work"
    logs = root / "logs"
    mineru_work = work / "mineru_api"
    mineru_logs = logs / "mineru_api"
    return RuntimePaths(
        runtime_root=root.resolve(),
        work_root=work.resolve(),
        logs_root=logs.resolve(),
        mineru_work_dir=mineru_work.resolve(),
        mineru_logs_dir=mineru_logs.resolve(),
    )

