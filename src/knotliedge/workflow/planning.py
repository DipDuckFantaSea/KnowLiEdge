from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Literal

from knotliedge.config.types import AppConfig
from knotliedge.workflow.artifacts import create_workflow_run_paths
from knotliedge.workflow.markdown import render_workflow_plan_markdown
from knotliedge.workflow.types import TaskIntent, WorkflowPlan, WorkflowStep

logger = logging.getLogger(__name__)

WorkflowPlanMode = Literal["fast", "standard"]


_ALLOWED_INTENTS: set[str] = {"research", "summary", "compare", "extract_table", "translate", "other"}


def normalize_intent(raw: str) -> TaskIntent:
    """Normalize a user-provided intent string into a supported TaskIntent.

    Args:
        raw: Raw intent string from CLI/MCP.

    Returns:
        A supported ``TaskIntent`` value, defaulting to ``other`` when unknown.
    """
    s = str(raw or "").strip().lower()
    if s in _ALLOWED_INTENTS:
        return s  # type: ignore[return-value]
    return "other"


def guess_research_steps(*, prompt: str, mode: WorkflowPlanMode) -> List[WorkflowStep]:
    """Heuristic router: build a replayable step list (M1 MVP).

    Args:
        prompt: User request.
        mode: Planning mode. ``fast`` omits optional graph steps unless hinted.

    Returns:
        Ordered workflow steps.
    """
    p = (prompt or "").strip()
    p_low = p.lower()

    steps: List[WorkflowStep] = []

    # Unified search first (always) to reduce cognitive load on the LLM side.
    steps.append(
        WorkflowStep(
            id="universal_search_local",
            tool="universal_academic_search",
            inputs={"query": p, "mode": "local", "fused": True, "top_k": 10},
            expected_output="Unified local search hits with chunk ids and source_md hints.",
            save_to="artifacts/universal_search_local.json",
        )
    )

    # Dual-track hybrid search (local deep KB + venue radar quarantine) by default.
    steps.append(
        WorkflowStep(
            id="universal_search_hybrid",
            tool="universal_academic_search",
            inputs={
                "query": p,
                "mode": "hybrid",
                "use_radar": True,
                "top_k_total": 20,
                "top_k_local_docs": 10,
                "top_k_radar_works": 10,
                "ai_parse_query": True,
            },
            expected_output="Unified hybrid payload wrapping dual-track local+radar search (includes markdown report).",
            save_to="artifacts/universal_search_hybrid.json",
        )
    )

    # Optional graph step: only in standard mode (or when explicitly hinted).
    graph_hint = any(k in p_low for k in ["演进", "谱系", "lineage", "evolution", "citation", "引用链"])
    if mode == "standard" or graph_hint:
        steps.append(
            WorkflowStep(
                id="trace_research_evolution",
                tool="trace_research_evolution",
                inputs={"start_id": "doc:<doc_id_or_openalex_work_id>", "end_id": "<target_id>"},
                expected_output="Markdown chain describing a shortest path between two nodes in citation graph.",
                save_to="artifacts/trace_research_evolution.md",
            )
        )

    # Expand context once chunk_ids are known.
    steps.append(
        WorkflowStep(
            id="expand_context_for_key_chunks",
            tool="get_knowledge_chunk",
            inputs={"chunk_id": "<chunk_id_from_search>", "window": 2},
            expected_output="Expanded context text for selected chunks to support grounded writing.",
            save_to="artifacts/key_chunks_context.md",
        )
    )

    # Ingestion guidance (best-effort; user may skip if vault is already complete).
    steps.append(
        WorkflowStep(
            id="optional_ingest_new_pdfs",
            tool="watch_ingest / pdf_to_md / index_markdown",
            inputs={"note": "Only if local KB is missing relevant papers."},
            expected_output="Updated vault + refreshed Chroma/FTS indexes for new PDFs.",
            save_to="artifacts/ingest_notes.md",
        )
    )

    return steps


def build_and_write_workflow_plan_markdown(
    *,
    cfg: AppConfig,
    prompt: str,
    intent: str,
    mode: WorkflowPlanMode,
) -> tuple[WorkflowPlan, Path]:
    """Create a workflow plan and write replayable artifacts under ``output/workflows/``.

    Args:
        cfg: Loaded application config (for ``project_root``).
        prompt: User request.
        intent: Raw intent string (normalized).
        mode: Planning mode.

    Returns:
        A tuple of ``(plan, plan_md_path)`` where ``plan_md_path`` points to the written
        Markdown plan on disk.
    """
    run_paths = create_workflow_run_paths(project_root=cfg.project_root)
    plan = WorkflowPlan(
        prompt=str(prompt),
        intent=normalize_intent(str(intent)),
        created_at=run_paths.run_dir.name,
        steps=guess_research_steps(prompt=str(prompt), mode=mode),
        notes=[
            "M1 scope: this tool returns a Markdown plan and writes replayable artifacts; it does not auto-execute steps yet.",
            "Follow steps in order; prefer fused local search before pulling long contexts.",
            "Stop early if the query scope is ambiguous (time range, device platform, metric definitions).",
        ],
    )

    md = render_workflow_plan_markdown(plan)
    run_paths.plan_md.write_text(md, encoding="utf-8")
    run_paths.run_jsonl.write_text(
        json.dumps({"event": "plan_generated", "created_at": plan.created_at, "intent": plan.intent, "mode": mode}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    logger.info("Workflow plan written: %s", run_paths.plan_md)
    return plan, run_paths.plan_md
