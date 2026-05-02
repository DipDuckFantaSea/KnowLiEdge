from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from knotliedge.executors.base import (
    EvidenceInput,
    ExecutorContext,
    ExecutorResult,
    create_executor_context,
    write_json,
    write_text,
)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def _clean_text_for_extract(text: str) -> str:
    t = str(text or "").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _extractive_bullets(text: str, *, max_bullets: int, max_chars_each: int) -> List[str]:
    t = _clean_text_for_extract(text)
    if not t:
        return []
    # Prefer sentence-like splits; fall back to lines.
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(t) if p.strip()]
    if len(parts) < 2:
        parts = [ln.strip() for ln in t.splitlines() if ln.strip()]
    out: List[str] = []
    for p in parts:
        p2 = re.sub(r"\s+", " ", p).strip()
        if not p2:
            continue
        if len(p2) > int(max_chars_each):
            p2 = p2[: int(max_chars_each)].rstrip() + "…"
        out.append(p2)
        if len(out) >= int(max_bullets):
            break
    return out


def run_summarize_with_citations(
    *,
    project_root: Path,
    evidence: EvidenceInput,
    get_chunk_context: Callable[[str, int], Dict[str, Any]],
    window: int = 2,
    max_bullets_per_chunk: int = 2,
    max_chars_each_bullet: int = 220,
    run_id: Optional[str] = None,
) -> ExecutorResult:
    """Build an evidence-grounded extractive summary with strict citations.

    This MVP never invents claims: each bullet comes from one chunk context and is
    cited via footnotes.
    """

    ctx: ExecutorContext = create_executor_context(project_root=Path(project_root), run_id=run_id)
    chunk_ids = evidence.normalized_chunk_ids()
    if not chunk_ids:
        return ExecutorResult(
            ok=False,
            message="No chunk_ids provided (need evidence to summarize).",
            missing_evidence=["chunk_ids"],
        )

    contexts: List[Dict[str, Any]] = []
    missing: List[str] = []
    for cid in chunk_ids:
        try:
            pack = get_chunk_context(str(cid), int(window))
            contexts.append(pack if isinstance(pack, dict) else {"chunk_id": cid, "text": str(pack)})
        except Exception as e:
            missing.append(f"{cid}: {type(e).__name__}: {e}")

    if not contexts:
        return ExecutorResult(
            ok=False,
            message="Failed to fetch any chunk context.",
            missing_evidence=missing or ["chunk_contexts"],
        )

    bullets: List[str] = []
    footnotes: List[str] = []
    evidence_map: List[Dict[str, Any]] = []
    for i, c in enumerate(contexts, start=1):
        cid = str(c.get("chunk_id") or f"chunk_{i}")
        source_md = str(c.get("source_md") or "")
        doc_id = str(c.get("doc_id") or "")
        short_name = str(c.get("short_name") or doc_id)
        text = str(c.get("text") or "")
        extracted = _extractive_bullets(text, max_bullets=int(max_bullets_per_chunk), max_chars_each=int(max_chars_each_bullet))
        if not extracted:
            extracted = ["NEED_EVIDENCE: empty/invalid chunk text"]
        note = f"[^{cid}]"
        for b in extracted:
            bullets.append(f"- {b} {note}")
            evidence_map.append(
                {
                    "bullet": b,
                    "citation": cid,
                    "chunk_id": cid,
                    "source_md": source_md,
                    "short_name": short_name,
                }
            )
        loc = f"source_md=`{source_md}`" if source_md else "source_md=N/A"
        footnotes.append(f"[^{cid}]: chunk_id=`{cid}`  short_name=`{short_name or 'N/A'}`  {loc}")

    md_lines: List[str] = []
    md_lines.append("# Summary (grounded)")
    md_lines.append("")
    md_lines.append("## Key points")
    md_lines.append("")
    md_lines.extend(bullets if bullets else ["- NEED_EVIDENCE: no usable content"])
    md_lines.append("")
    md_lines.append("## Citations")
    md_lines.append("")
    md_lines.extend(footnotes)
    md_lines.append("")

    out_md = "\n".join(md_lines).rstrip() + "\n"
    md_path = write_text(ctx.paths.artifacts_dir / "summary_with_citations.md", out_md)
    ev_path = write_json(
        ctx.paths.artifacts_dir / "summary_with_citations.evidence.json",
        {
            "run_id": ctx.run_id,
            "window": int(window),
            "chunk_ids": chunk_ids,
            "missing": missing,
            "bullets": evidence_map,
        },
    )
    ctx.write_jsonl_event({"event": "executor_done", "executor": "summarize_with_citations", "ok": True})

    preview = "\n".join(md_lines[: min(len(md_lines), 30)]).rstrip() + "\n"
    msg = f"Summary written: {md_path}"
    if missing:
        msg += f" (missing {len(missing)} chunks)"
    return ExecutorResult(
        ok=True,
        message=msg,
        artifacts={
            "summary_md": str(md_path),
            "evidence_json": str(ev_path),
            "run_dir": str(ctx.paths.run_dir),
        },
        preview_markdown=preview,
        missing_evidence=missing,
    )

