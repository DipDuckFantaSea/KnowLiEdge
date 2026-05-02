from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@dataclass(frozen=True)
class WorkflowRunPaths:
    """Standardized artifact layout for a workflow run under `output/workflows/`.

    Layout:
        output/workflows/<ts>/
          plan.md
          run.jsonl
          artifacts/
    """

    run_dir: Path
    plan_md: Path
    run_jsonl: Path
    artifacts_dir: Path


def create_workflow_run_paths(*, project_root: Path, created_at: Optional[str] = None) -> WorkflowRunPaths:
    """Create (and ensure) output paths for a workflow run.

    Args:
        project_root: Repository/project root directory.
        created_at: Optional UTC timestamp string. If omitted, use current time.

    Returns:
        A `WorkflowRunPaths` object with resolved paths.
    """

    ts = str(created_at or _now_ts()).strip() or _now_ts()
    root = Path(project_root).resolve() / "output" / "workflows" / ts
    artifacts = root / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return WorkflowRunPaths(
        run_dir=root.resolve(),
        plan_md=(root / "plan.md").resolve(),
        run_jsonl=(root / "run.jsonl").resolve(),
        artifacts_dir=artifacts.resolve(),
    )

