from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from knotliedge.citation_graph.manager import CitationGraphManager
from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenAlex citation subgraph to interactive HTML (Pyvis).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--db", type=str, default=None, help="Override citations.sqlite3 path.")
    parser.add_argument(
        "--include-doc-bridge",
        action="store_true",
        help="Include synthetic doc:* nodes when FTS links exist.",
    )
    parser.add_argument("--center-doc-id", type=str, default=None, help="Optional local doc_id to center the subgraph.")
    parser.add_argument("--center-work-id", type=str, default=None, help="Optional OpenAlex work id/url to center the subgraph.")
    parser.add_argument("--radius", type=int, default=2, help="Ego graph radius (undirected) when center is set.")
    parser.add_argument("--max-nodes", type=int, default=1200, help="Max nodes to keep after centering (best-effort).")
    args = parser.parse_args()

    cfg = load_app_config(Path(args.config))
    cite_db = Path(args.db) if args.db else None
    mgr = CitationGraphManager(cfg=cfg, citation_db_path=cite_db)
    include_bridge = bool(args.include_doc_bridge) or bool(args.center_doc_id)
    g = mgr.load_graph(include_local_doc_bridge=include_bridge)

    center = None
    if args.center_doc_id:
        center = f"doc:{str(args.center_doc_id).strip()}"
        if center not in g:
            # Fallback: if bridge node missing, use the linked work id.
            center = mgr._resolve_doc_to_work_id(str(args.center_doc_id).strip())[0]  # type: ignore[attr-defined]
    elif args.center_work_id:
        from knotliedge.citation_graph.openalex_store import normalize_openalex_work_id

        center = normalize_openalex_work_id(str(args.center_work_id).strip())

    if center and center in g:
        import networkx as nx

        ug = g.to_undirected()
        ego = nx.ego_graph(ug, center, radius=int(args.radius))
        if ego.number_of_nodes() > int(args.max_nodes):
            # Keep highest-degree nodes as a rough cap to avoid huge HTMLs.
            deg = sorted(ego.degree, key=lambda x: x[1], reverse=True)
            keep = {n for n, _d in deg[: int(args.max_nodes)]}
            keep.add(center)
            ego = ego.subgraph(keep).copy()
        g = g.subgraph(ego.nodes()).copy()
        # Preserve edge directions where possible.

    try:
        from pyvis.network import Network
    except Exception as e:
        raise RuntimeError("pyvis is required. Install with: conda run -n agent python -m pip install pyvis") from e

    net = Network(height="720px", width="100%", directed=True, bgcolor="#111111", font_color="#eeeeee")
    for nid, attrs in g.nodes(data=True):
        label = str(attrs.get("short_name") or attrs.get("title") or nid)
        title = str(attrs.get("title") or "")
        year = attrs.get("year")
        extra = f"{title}\n{year}" if title else str(nid)
        is_center = bool(center) and str(nid) == str(center)
        net.add_node(
            str(nid),
            label=label[:80],
            title=extra[:500],
            color="#ffcc00" if is_center else None,
        )

    for u, v, _edata in g.edges(data=True):
        net.add_edge(str(u), str(v))

    out_dir = (cfg.project_root / "output" / "viz").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = "citation_graph"
    if args.center_doc_id:
        stem = f"citation_graph_doc_{str(args.center_doc_id).strip()}"
    elif args.center_work_id:
        stem = f"citation_graph_work_{str(args.center_work_id).strip().replace(':','_').replace('/','_')}"
    out_path = out_dir / f"{stem}_{ts}.html"
    net.write_html(str(out_path))
    logger.info("Wrote visualization | path=%s | nodes=%s edges=%s", out_path, g.number_of_nodes(), g.number_of_edges())


if __name__ == "__main__":
    main()
