from __future__ import annotations

import argparse
import sys
from pathlib import Path

from knotliedge.config.load import load_app_config
from knotliedge.embeddings import get_embedder
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.ops.chroma_health import smoke_search_hits, summarize_chroma_collection
from knotliedge.storage.chroma_store import ChromaStore


logger = setup_logging()


def _render_rich_verify(*, summary: dict, query: str, hits: list) -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except Exception:
        return

    console = Console(
        stderr=True,
        force_terminal=True,
        legacy_windows=True,
        emoji=False,
        highlight=False,
        markup=False,
        safe_box=True,
    )

    table = Table(title="Chroma collection summary", show_lines=False, box=box.ASCII)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    for k in ["collection_name", "collection_count", "unique_doc_ids", "latest_created_at"]:
        v = summary.get(k)
        table.add_row(str(k), "" if v is None else str(v))
    if summary.get("scan_error"):
        table.add_row("scan_error", str(summary.get("scan_error")))

    console.print(
        Panel.fit(
            table,
            title="KnotLiEdge / verify_chroma",
            border_style="blue",
            box=box.ASCII2,
        )
    )

    hits_table = Table(title=f"Smoke search hits (query={query!r})", show_lines=False, box=box.ASCII)
    hits_table.add_column("#", justify="right", style="magenta", no_wrap=True)
    hits_table.add_column("score", justify="right", style="green")
    hits_table.add_column("doc_id", style="cyan", overflow="fold")
    hits_table.add_column("short_name", style="cyan", overflow="fold")
    hits_table.add_column("section", style="yellow", overflow="fold")
    hits_table.add_column("preview", style="white", overflow="fold")

    for i, h in enumerate(hits, start=1):
        hits_table.add_row(
            str(i),
            "" if h.get("score") is None else f"{float(h.get('score')):.6f}",
            str(h.get("doc_id") or ""),
            str(h.get("short_name") or ""),
            str(h.get("section") or ""),
            str(h.get("preview") or ""),
        )

    console.print(
        Panel.fit(
            hits_table,
            title="Vector search",
            border_style="green",
            box=box.ASCII2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ChromaDB has indexed chunks and is searchable.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--query", type=str, default="calibration", help="Search query.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k hits.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_app_config(cfg_path)
    try:
        embedder = get_embedder(config_path=cfg_path)
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)

    store = ChromaStore(cfg=cfg, embedder=embedder)
    summary = summarize_chroma_collection(store)
    logger.info(
        "collection=%s count=%s unique_doc_ids=%s latest_created_at=%s",
        summary.get("collection_name"),
        summary.get("collection_count"),
        summary.get("unique_doc_ids"),
        summary.get("latest_created_at"),
    )
    if summary.get("scan_error"):
        logger.error("metadata scan error: %s", summary["scan_error"])

    hits = smoke_search_hits(store, str(args.query), top_k=int(args.top_k))
    logger.info("query=%s hits=%s", args.query, len(hits))
    for i, h in enumerate(hits, start=1):
        logger.info(
            "hit#%s score=%s doc_id=%s short=%s section=%s",
            i,
            h.get("score"),
            h.get("doc_id"),
            h.get("short_name"),
            h.get("section"),
        )
        logger.info("hit#%s preview=%s", i, h.get("preview") or "")

    _render_rich_verify(summary=summary, query=str(args.query), hits=hits)


if __name__ == "__main__":
    main()
