"""Workflow protocol (Planner/Router foundation).

This package defines a minimal, tool-agnostic plan representation for research workflows.
M0 scope: define types + artifact layout + markdown rendering, without executing steps.
"""

from knotliedge.workflow.artifacts import WorkflowRunPaths, create_workflow_run_paths
from knotliedge.workflow.markdown import render_workflow_plan_markdown
from knotliedge.workflow.planning import build_and_write_workflow_plan_markdown, guess_research_steps, normalize_intent
from knotliedge.workflow.types import (
    EvidenceRef,
    EvidenceRefKind,
    TaskIntent,
    WorkflowPlan,
    WorkflowStep,
)

__all__ = [
    "WorkflowRunPaths",
    "create_workflow_run_paths",
    "render_workflow_plan_markdown",
    "build_and_write_workflow_plan_markdown",
    "guess_research_steps",
    "normalize_intent",
    "EvidenceRef",
    "EvidenceRefKind",
    "TaskIntent",
    "WorkflowPlan",
    "WorkflowStep",
]

