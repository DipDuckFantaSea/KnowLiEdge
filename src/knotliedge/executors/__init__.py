from __future__ import annotations

"""Single-responsibility executors for replayable research workflows.

Executors are deliberately narrow: they transform a bounded set of evidence into
auditable artifacts under ``output/workflows/<run_id>/artifacts/`` and return a
small summary for MCP/CLI display.
"""

from knotliedge.executors.base import (
    ExecutorContext,
    ExecutorResult,
    EvidenceInput,
    create_executor_context,
)

__all__ = [
    "ExecutorContext",
    "ExecutorResult",
    "EvidenceInput",
    "create_executor_context",
]

