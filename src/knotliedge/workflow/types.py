from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


TaskIntent = Literal[
    "research",
    "summary",
    "compare",
    "extract_table",
    "translate",
    "other",
]

EvidenceRefKind = Literal[
    "local_chunk",
    "openalex_work",
    "citation_node",
]


@dataclass(frozen=True)
class EvidenceRef:
    """A minimal evidence pointer that can be audited later.

    This is intentionally tool-agnostic and supports multiple backends.

    Args:
        kind: Evidence kind discriminator.
        label: Human-readable label shown in Markdown plans.
        chunk_id: For local KB chunks (`get_knowledge_chunk`).
        source_md: Absolute path to markdown file (when available).
        openalex_work_id: OpenAlex work URL/id.
        citation_node_id: Node id used by citation graph manager/store.
        note: Optional free-form note (e.g., why this evidence is relevant).
    """

    kind: EvidenceRefKind
    label: str
    chunk_id: Optional[str] = None
    source_md: Optional[str] = None
    openalex_work_id: Optional[str] = None
    citation_node_id: Optional[str] = None
    note: Optional[str] = None

    def to_compact_id(self) -> str:
        """Return a stable-ish compact identifier for plan rendering."""
        if self.kind == "local_chunk":
            return str(self.chunk_id or "unknown_chunk")
        if self.kind == "openalex_work":
            return str(self.openalex_work_id or "unknown_work")
        if self.kind == "citation_node":
            return str(self.citation_node_id or "unknown_node")
        return "unknown"


@dataclass(frozen=True)
class WorkflowStep:
    """A single step in a workflow plan.

    Args:
        id: Step id (unique within the plan).
        tool: Callable capability name (e.g. MCP tool name, or script entry).
        inputs: Serializable inputs required by the tool.
        expected_output: Human-readable expectation of what this step produces.
        save_to: Relative path under run artifacts dir, or empty to skip saving.
    """

    id: str
    tool: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    save_to: str = ""


@dataclass(frozen=True)
class WorkflowPlan:
    """A replayable workflow plan (M0 protocol).

    Args:
        prompt: Original user request.
        intent: High-level intent category.
        created_at: UTC timestamp string (YYYYmmdd-HHMMSS).
        steps: Ordered steps to run.
        evidence: Optional initial evidence refs (can be empty for pure planning).
        notes: Extra guidance, stop conditions, or constraints.
    """

    prompt: str
    intent: TaskIntent = "other"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    steps: List[WorkflowStep] = field(default_factory=list)
    evidence: List[EvidenceRef] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

