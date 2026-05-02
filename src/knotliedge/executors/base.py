from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from knotliedge.workflow.artifacts import WorkflowRunPaths, create_workflow_run_paths
from knotliedge.workflow.types import EvidenceRef


@dataclass(frozen=True)
class EvidenceInput:
    """Executor evidence inputs.

    Executors should prefer chunk-based grounding for auditability.

    Args:
        chunk_ids: Local KB chunk ids (for ``get_knowledge_chunk``).
        evidence_refs: Optional structured refs; can include local_chunk/openalex/citation.
    """

    chunk_ids: List[str] = field(default_factory=list)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)

    def normalized_chunk_ids(self) -> List[str]:
        ids = [str(x).strip() for x in (self.chunk_ids or []) if str(x).strip()]
        # EvidenceRef(local_chunk) can also carry chunk ids.
        for e in self.evidence_refs or []:
            if getattr(e, "kind", None) == "local_chunk" and getattr(e, "chunk_id", None):
                ids.append(str(e.chunk_id).strip())
        # stable de-dup
        seen: set[str] = set()
        out: List[str] = []
        for cid in ids:
            if cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
        return out


@dataclass(frozen=True)
class ExecutorContext:
    """Execution context for writing auditable artifacts."""

    project_root: Path
    run_id: str
    paths: WorkflowRunPaths

    def artifact_path(self, rel: str) -> Path:
        r = str(rel or "").lstrip("/").replace("\\", "/")
        return (self.paths.run_dir / r).resolve()

    def write_jsonl_event(self, event: Dict[str, Any]) -> None:
        obj = dict(event or {})
        obj.setdefault("run_id", self.run_id)
        self.paths.run_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.run_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class ExecutorResult:
    """Minimal executor output for MCP/CLI.

    Args:
        ok: Whether the executor produced usable artifacts.
        message: Short status message.
        artifacts: Mapping from logical name to absolute file path.
        preview_markdown: Optional short markdown snippet to show inline.
        missing_evidence: If non-empty, executor is signaling evidence gaps.
    """

    ok: bool
    message: str
    artifacts: Dict[str, str] = field(default_factory=dict)
    preview_markdown: str = ""
    missing_evidence: List[str] = field(default_factory=list)


def create_executor_context(*, project_root: Path, run_id: Optional[str] = None) -> ExecutorContext:
    """Create a standardized executor run directory under ``output/workflows``."""

    paths = create_workflow_run_paths(project_root=Path(project_root), created_at=run_id)
    return ExecutorContext(project_root=Path(project_root).resolve(), run_id=paths.run_dir.name, paths=paths)


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")
    return path.resolve()


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.resolve()


def write_csv_rows(path: Path, *, header: Sequence[str], rows: Sequence[Sequence[str]]) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow([str(x) for x in r])
    return path.resolve()

