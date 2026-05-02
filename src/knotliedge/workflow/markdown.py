from __future__ import annotations

import json
from typing import List

from knotliedge.workflow.types import EvidenceRef, WorkflowPlan, WorkflowStep


def _md_escape(text: str) -> str:
    s = str(text or "")
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _render_evidence(evs: List[EvidenceRef]) -> str:
    if not evs:
        return ""
    lines: List[str] = ["## Evidence (initial)", ""]
    for i, e in enumerate(evs, start=1):
        parts: List[str] = [f"{i}. **{_md_escape(e.label)}**", f"- kind: `{e.kind}`", f"- id: `{_md_escape(e.to_compact_id())}`"]
        if e.source_md:
            parts.append(f"- source_md: `{_md_escape(e.source_md)}`")
        if e.note:
            parts.append(f"- note: {_md_escape(e.note)}")
        lines.extend(parts)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_step(step: WorkflowStep, idx: int) -> str:
    lines: List[str] = [
        f"{idx}. **{_md_escape(step.id)}**",
        f"   - tool: `{_md_escape(step.tool)}`",
    ]
    if step.inputs:
        payload = json.dumps(step.inputs, ensure_ascii=False, indent=2)
        lines.append("   - inputs:")
        lines.append("")
        lines.append("```json")
        lines.append(payload)
        lines.append("```")
    if step.expected_output:
        lines.append(f"   - expected_output: {_md_escape(step.expected_output)}")
    if step.save_to:
        lines.append(f"   - save_to: `{_md_escape(step.save_to)}`")
    return "\n".join(lines)


def render_workflow_plan_markdown(plan: WorkflowPlan) -> str:
    """Render a workflow plan into a standard Markdown document (M0)."""
    lines: List[str] = []
    lines.append("# Workflow Plan")
    lines.append("")
    lines.append(f"- created_at: `{_md_escape(plan.created_at)}` (UTC)")
    lines.append(f"- intent: `{_md_escape(plan.intent)}`")
    lines.append("")
    lines.append("## Prompt")
    lines.append("")
    lines.append(_md_escape(plan.prompt).strip())
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    if not plan.steps:
        lines.append("_No steps yet (planning only)._")
        lines.append("")
    else:
        for i, st in enumerate(plan.steps, start=1):
            lines.append(_render_step(st, i))
            lines.append("")

    ev_md = _render_evidence(plan.evidence)
    if ev_md:
        lines.append(ev_md.rstrip())
        lines.append("")

    if plan.notes:
        lines.append("## Notes / Stop conditions")
        lines.append("")
        for n in plan.notes:
            lines.append(f"- {_md_escape(n).strip()}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"

