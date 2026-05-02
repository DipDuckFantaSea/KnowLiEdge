from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from knotliedge.executors.base import (
    ExecutorContext,
    ExecutorResult,
    create_executor_context,
    write_csv_rows,
    write_json,
    write_text,
)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def _first_sentence(text: str, *, max_chars: int) -> str:
    t = str(text or "").replace("\r", "\n").strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(t) if p.strip()]
    s = parts[0] if parts else t
    if len(s) > int(max_chars):
        s = s[: int(max_chars)].rstrip() + "…"
    return s


def _md_escape_pipe(s: str) -> str:
    return str(s or "").replace("|", "\\|").replace("\n", " ").strip()


def run_compare_papers_by_fields(
    *,
    project_root: Path,
    papers: Sequence[Dict[str, Any]],
    fields: Optional[Sequence[str]],
    get_chunk_context: Callable[[str, int], Dict[str, Any]],
    window: int = 1,
    max_chars_each_cell: int = 220,
    run_id: Optional[str] = None,
) -> ExecutorResult:
    """Compare multiple papers by a fixed set of fields (MVP, evidence-grounded).

    Each paper is represented by a small evidence set (chunk ids). This MVP is
    conservative: each cell uses extractive snippets from evidence contexts and
    attaches strict footnote citations.
    """

    ctx: ExecutorContext = create_executor_context(project_root=Path(project_root), run_id=run_id)
    use_fields = [str(x).strip() for x in (fields or []) if str(x).strip()] or [
        "工艺",
        "材料/结构",
        "指标/数据",
        "方法",
        "结论",
        "局限",
    ]
    ps = [p for p in (papers or []) if isinstance(p, dict)]
    if len(ps) < 2:
        return ExecutorResult(ok=False, message="Need at least 2 papers to compare.", missing_evidence=["papers>=2"])

    # Resolve per-paper label and evidence chunk_ids
    paper_specs: List[Tuple[str, List[str]]] = []
    for i, p in enumerate(ps, start=1):
        label = str(p.get("label") or p.get("short_name") or p.get("doc_id") or f"paper_{i}").strip()
        cids_raw = p.get("chunk_ids") or p.get("chunks") or []
        cids = [str(x).strip() for x in cids_raw if str(x).strip()] if isinstance(cids_raw, list) else []
        if not cids:
            continue
        paper_specs.append((label, cids))

    if len(paper_specs) < 2:
        return ExecutorResult(
            ok=False,
            message="Need at least 2 papers with non-empty chunk_ids.",
            missing_evidence=["chunk_ids"],
        )

    # Fetch contexts
    footnotes: Dict[str, str] = {}
    cell_map: Dict[str, Dict[str, Dict[str, Any]]] = {f: {} for f in use_fields}
    missing: List[str] = []

    for label, cids in paper_specs:
        # Prefer first chunk as primary citation, but also scan others for snippet variety.
        contexts: List[Dict[str, Any]] = []
        for cid in cids:
            try:
                contexts.append(get_chunk_context(str(cid), int(window)))
            except Exception as e:
                missing.append(f"{label}:{cid}: {type(e).__name__}: {e}")
        if not contexts:
            for f in use_fields:
                cell_map[f][label] = {"text": "NEED_EVIDENCE", "citations": []}
            continue

        # Use a stable citation id per chunk
        for c in contexts:
            cid = str(c.get("chunk_id") or "")
            if cid and cid not in footnotes:
                source_md = str(c.get("source_md") or "")
                short_name = str(c.get("short_name") or "")
                loc = f"source_md=`{source_md}`" if source_md else "source_md=N/A"
                footnotes[cid] = f"[^{cid}]: chunk_id=`{cid}`  short_name=`{short_name or label}`  {loc}"

        # MVP: use first sentence of the best context for every field (conservative).
        # Later iterations can add field-specific extraction.
        primary = contexts[0]
        primary_text = _first_sentence(str(primary.get("text") or ""), max_chars=int(max_chars_each_cell))
        primary_cid = str(primary.get("chunk_id") or "")
        if not primary_text:
            primary_text = "NEED_EVIDENCE"

        for f in use_fields:
            cell_map[f][label] = {
                "text": primary_text,
                "citations": [primary_cid] if primary_cid else [],
            }

    # Render Markdown table
    labels = [lbl for lbl, _ in paper_specs]
    header = ["字段", *labels]
    md_lines: List[str] = []
    md_lines.append("# Paper comparison (grounded)")
    md_lines.append("")
    md_lines.append("| " + " | ".join(_md_escape_pipe(x) for x in header) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    csv_rows: List[List[str]] = []
    for f in use_fields:
        row_md: List[str] = [_md_escape_pipe(f)]
        row_csv: List[str] = [f]
        for lbl in labels:
            pack = cell_map.get(f, {}).get(lbl, {}) if isinstance(cell_map.get(f, {}), dict) else {}
            txt = str(pack.get("text") or "").strip()
            cits = pack.get("citations") if isinstance(pack.get("citations"), list) else []
            cit_note = ""
            if cits:
                cit_note = " " + " ".join([f"[^{str(c).strip()}]" for c in cits if str(c).strip()])
            cell_md = _md_escape_pipe(txt + cit_note) if txt else "NEED_EVIDENCE"
            row_md.append(cell_md)
            row_csv.append(txt)
        md_lines.append("| " + " | ".join(row_md) + " |")
        csv_rows.append(row_csv)

    md_lines.append("")
    md_lines.append("## Citations")
    md_lines.append("")
    for cid in sorted(footnotes.keys()):
        md_lines.append(footnotes[cid])
    md_lines.append("")

    out_md = "\n".join(md_lines).rstrip() + "\n"
    md_path = write_text(ctx.paths.artifacts_dir / "compare_papers_by_fields.md", out_md)
    csv_path = write_csv_rows(
        ctx.paths.artifacts_dir / "compare_papers_by_fields.csv",
        header=header,
        rows=csv_rows,
    )
    ev_path = write_json(
        ctx.paths.artifacts_dir / "compare_papers_by_fields.evidence.json",
        {
            "run_id": ctx.run_id,
            "window": int(window),
            "fields": use_fields,
            "papers": [{"label": lbl, "chunk_ids": cids} for lbl, cids in paper_specs],
            "cells": cell_map,
            "missing": missing,
        },
    )
    ctx.write_jsonl_event({"event": "executor_done", "executor": "compare_papers_by_fields", "ok": True})

    preview = "\n".join(md_lines[: min(len(md_lines), 40)]).rstrip() + "\n"
    msg = f"Comparison written: {md_path}"
    if missing:
        msg += f" (missing {len(missing)} chunks)"
    return ExecutorResult(
        ok=True,
        message=msg,
        artifacts={
            "compare_md": str(md_path),
            "compare_csv": str(csv_path),
            "evidence_json": str(ev_path),
            "run_dir": str(ctx.paths.run_dir),
        },
        preview_markdown=preview,
        missing_evidence=missing,
    )

