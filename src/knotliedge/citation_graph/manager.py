from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from knotliedge.citation_graph.openalex_store import OpenAlexCitationStore, normalize_openalex_work_id
from knotliedge.citation_graph.store import default_citation_db_path
from knotliedge.config.load import load_app_config
from knotliedge.config.types import AppConfig
from knotliedge.storage.fts_store import default_fts_db_path

logger = logging.getLogger(__name__)


def _short_name_from_title(title: str, *, year: Optional[int]) -> str:
    """Build a compact label for MCP / LLM context (not vault short_name)."""
    t = (title or "").strip()
    if len(t) > 48:
        t = t[:45].rstrip() + "…"
    if year is not None:
        return f"{t} ({year})" if t else str(year)
    return t or "untitled"


def _doc_short_name_from_fts(fts_db: Path, doc_id: str) -> Optional[str]:
    """Best-effort ``short_name`` for a vault ``doc_id`` from FTS sidecar."""
    p = Path(fts_db).resolve()
    if not p.is_file():
        return None
    try:
        con = sqlite3.connect(str(p))
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT short_name FROM chunks_fts WHERE doc_id = ? LIMIT 1;",
            (str(doc_id),),
        ).fetchone()
        con.close()
    except Exception as e:
        logger.warning("FTS short_name lookup failed | doc_id=%s | %s", doc_id, e)
        return None
    if row is None:
        return None
    sn = str(row["short_name"] or "").strip()
    return sn or None


def _doc_openalex_and_title(fts_db: Path, doc_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(openalex_id, openalex_title)`` from FTS ``documents`` table."""
    p = Path(fts_db).resolve()
    if not p.is_file():
        return None, None
    try:
        con = sqlite3.connect(str(p))
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT openalex_id, openalex_title FROM documents WHERE doc_id = ? LIMIT 1;",
            (str(doc_id),),
        ).fetchone()
        con.close()
    except Exception as e:
        logger.warning("FTS documents lookup failed | doc_id=%s | %s", doc_id, e)
        return None, None
    if row is None:
        return None, None
    oa = str(row["openalex_id"] or "").strip() or None
    tt = str(row["openalex_title"] or "").strip() or None
    return oa, tt


class CitationGraphManager:
    """NetworkX-backed views over OpenAlex work nodes stored in the citation SQLite DB."""

    def __init__(
        self,
        *,
        cfg: AppConfig,
        citation_db_path: Optional[Path] = None,
    ) -> None:
        """Args:
            cfg: Loaded application config (paths resolved).
            citation_db_path: Override SQLite path; default uses ``default_citation_db_path``.
        """
        self._cfg = cfg
        db = Path(citation_db_path) if citation_db_path is not None else default_citation_db_path(cfg)
        self._oa_store = OpenAlexCitationStore(db_path=db)
        self._fts_db = default_fts_db_path(cfg)

    @property
    def oa_store(self) -> OpenAlexCitationStore:
        return self._oa_store

    def load_graph(self, *, include_local_doc_bridge: bool = False) -> nx.DiGraph:
        """Build a directed graph: edge ``u -> v`` means work ``u`` cites work ``v``.

        Args:
            include_local_doc_bridge: If True, add synthetic nodes ``doc:{doc_id}`` linked
                to the work node for each FTS ``documents.openalex_id`` row.

        Returns:
            A :class:`networkx.DiGraph` with node attributes ``title``, ``year``, ``doi``,
            ``short_name``, ``kind`` (``work`` or ``local_doc``).
        """
        g = nx.DiGraph()
        for w in self._oa_store.iter_works():
            wid = str(w.get("work_id") or "")
            if not wid:
                continue
            title = str(w.get("title") or "")
            year = w.get("publication_year")
            yi: Optional[int] = int(year) if isinstance(year, int) else None
            g.add_node(
                wid,
                kind="work",
                title=title,
                year=yi,
                doi=str(w.get("doi") or "") or None,
                short_name=_short_name_from_title(title, year=yi),
            )
        for s, d, _src in self._oa_store.iter_cite_edges():
            if s not in g:
                g.add_node(s, kind="work", title="", year=None, doi=None, short_name=s)
            if d not in g:
                g.add_node(d, kind="work", title="", year=None, doi=None, short_name=d)
            g.add_edge(s, d)

        if include_local_doc_bridge:
            self._attach_local_doc_bridges(g)
        return g

    def _attach_local_doc_bridges(self, g: nx.DiGraph) -> None:
        p = self._fts_db
        if not p.is_file():
            return
        try:
            con = sqlite3.connect(str(p))
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT doc_id, openalex_id, openalex_title, publication_year FROM documents WHERE openalex_id IS NOT NULL AND trim(openalex_id) != '';"
            ).fetchall()
            con.close()
        except Exception as e:
            logger.warning("Failed to list documents for doc bridge | %s", e)
            return
        for row in rows:
            did = str(row["doc_id"] or "").strip()
            raw_oa = str(row["openalex_id"] or "").strip()
            wid = normalize_openalex_work_id(raw_oa)
            if not did or not wid:
                continue
            title = str(row["openalex_title"] or "").strip()
            py = row["publication_year"]
            yi: Optional[int] = int(py) if isinstance(py, int) else None
            sn = _doc_short_name_from_fts(p, did) or _short_name_from_title(title, year=yi)
            node_doc = f"doc:{did}"
            g.add_node(node_doc, kind="local_doc", title=title, year=yi, doi=None, short_name=sn)
            if wid not in g:
                g.add_node(wid, kind="work", title=title, year=yi, doi=None, short_name=sn)
            g.add_edge(node_doc, wid, bridge=True)

    def _resolve_doc_to_work_id(self, doc_id: str) -> Tuple[Optional[str], str]:
        """Return ``(normalized_work_id, human_note)``."""
        oa, title = _doc_openalex_and_title(self._fts_db, doc_id)
        if not oa:
            return None, "该 ``doc_id`` 在 FTS ``documents`` 中无 ``openalex_id``；请先运行元数据富化或扩展脚本。"
        wid = normalize_openalex_work_id(oa)
        if not wid:
            return None, f"无法解析 OpenAlex id: {oa}"
        _ = title
        return wid, ""

    def get_lineage(self, doc_id: str) -> str:
        """Return Markdown summarizing one-hop references (parents) and citing works (children).

        Args:
            doc_id: Vault document id (``documents.doc_id``).

        Returns:
            Markdown string suitable for LLM consumption (no large JSON).
        """
        wid, note = self._resolve_doc_to_work_id(doc_id)
        lines: List[str] = []
        sn_doc = _doc_short_name_from_fts(self._fts_db, doc_id)
        head = f"## 文献血缘（1 跳）\n\n**doc_id**: `{doc_id}`"
        if sn_doc:
            head += f"\n**short_name**: {sn_doc}"
        lines.append(head)
        if not wid:
            lines.append("")
            lines.append(note)
            return "\n".join(lines)

        lines.append(f"\n**OpenAlex work**: `{wid}`\n")

        wrow = self._oa_store.get_work(wid)
        if wrow:
            t = str(wrow.get("title") or "").strip()
            y = wrow.get("publication_year")
            ys = str(y) if isinstance(y, int) else "?"
            lines.append(f"- **本文标题**: {t} ({ys})")

        parents = self._oa_store.cite_successors(wid, limit=200)[:50]

        lines.append("\n### 引用（父节点，本文引用的工作）\n")
        if not parents:
            lines.append("- （本地库中暂无 OpenAlex 扩展边，或该文献无 ``referenced_works`` 记录）")
        else:
            for pid in parents:
                meta = self._oa_store.get_work(pid) or {}
                title = str(meta.get("title") or "").strip() or pid
                py = meta.get("publication_year")
                yi = int(py) if isinstance(py, int) else None
                psn = _short_name_from_title(title, year=yi)
                lines.append(f"- **{psn}**  (`{pid}`)")

        children = self._oa_store.cite_predecessors(wid, limit=200)[:50]

        lines.append("\n### 被引（子节点，引用本文的工作，可能截断）\n")
        if not children:
            lines.append("- （暂无记录；扩展脚本对被引方向有上限）")
        else:
            for cid in children:
                meta = self._oa_store.get_work(cid) or {}
                title = str(meta.get("title") or "").strip() or cid
                py = meta.get("publication_year")
                yi = int(py) if isinstance(py, int) else None
                csn = _short_name_from_title(title, year=yi)
                lines.append(f"- **{csn}**  (`{cid}`)")

        lines.append(
            "\n*提示*: 需要正文片段时请用 ``search_knowledge_base`` / ``get_knowledge_chunk``；"
            "雷达摘要集合与 ``openalex_works.abstract`` 可能另存。*"
        )
        return "\n".join(lines)

    def find_evolution_path(self, start_id: str, end_id: str) -> str:
        """Shortest chain between two works in the **undirected** citation graph.

        Args:
            start_id: OpenAlex work URL/id, or ``doc:{doc_id}`` when bridge graph was loaded manually.
            end_id: Same as ``start_id``.

        Returns:
            Markdown description of the path.
        """
        g = self.load_graph(include_local_doc_bridge=True)
        a = self._normalize_path_endpoint(start_id, g)
        b = self._normalize_path_endpoint(end_id, g)
        lines: List[str] = ["## 演进路径（无向最短链）\n"]
        if a is None:
            lines.append(f"- 无法解析起点: `{start_id}`")
        if b is None:
            lines.append(f"- 无法解析终点: `{end_id}`")
        if a is None or b is None:
            return "\n".join(lines)
        ug = g.to_undirected()
        try:
            path = nx.shortest_path(ug, source=a, target=b)
        except nx.NetworkXNoPath:
            return "\n".join(lines + [f"\n在已加载子图中 **无路径** 连接 `{a}` 与 `{b}`。\n"])
        except nx.NodeNotFound as e:
            return "\n".join(lines + [f"\n节点不存在于图中: {e}\n"])

        chain_parts: List[str] = []
        for nid in path:
            attrs = g.nodes.get(nid, {})
            sn = str(attrs.get("short_name") or nid)
            chain_parts.append(f"[{sn}]({nid})")
        lines.append(" -> ".join(chain_parts))
        lines.append("")
        lines.append("*说明*: 链基于无向化引用图的最短路径；步间语义占位，后续可用摘要细化。*")
        return "\n".join(lines)

    def _normalize_path_endpoint(self, raw: str, g: nx.DiGraph) -> Optional[str]:
        s = (raw or "").strip()
        if not s:
            return None
        if s.startswith("doc:"):
            node = s if s in g else None
            if node:
                return node
            did = s[len("doc:") :].strip()
            wid, _ = self._resolve_doc_to_work_id(did)
            return wid
        return normalize_openalex_work_id(s)

    def get_key_nodes(self, top_n: int = 5, *, center_work_id: Optional[str] = None) -> str:
        """Run PageRank on the full OA subgraph or on a bounded ego network.

        Args:
            top_n: Number of top nodes to list.
            center_work_id: If set, restrict to nodes within 2 hops (undirected) of this work.

        Returns:
            Markdown bullet list of key works with ``short_name`` and ``work_id``.
        """
        g = self.load_graph(include_local_doc_bridge=False)
        if g.number_of_nodes() == 0:
            return "## 关键节点（PageRank）\n\n（图为空：请先运行 ``expand_citation_network``。）\n"

        sub = g
        cwid = normalize_openalex_work_id(center_work_id) if center_work_id else None
        if cwid and cwid in g:
            ug = g.to_undirected()
            ego = nx.ego_graph(ug, cwid, radius=2)
            sub = g.subgraph(ego.nodes()).copy()

        pr = nx.pagerank(sub, alpha=0.9)
        ranked = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[: int(top_n)]
        lines: List[str] = ["## 关键节点（PageRank）\n"]
        if cwid:
            lines.append(f"*子图中心*: `{cwid}` （2-hop 无向邻居）\n")
        for rank, (nid, score) in enumerate(ranked, start=1):
            attrs = sub.nodes.get(nid, {})
            sn = str(attrs.get("short_name") or nid)
            lines.append(f"{rank}. **{sn}** — `{nid}` — score={score:.6f}")
        return "\n".join(lines) + "\n"


def build_manager_from_config_path(config_path: Path) -> CitationGraphManager:
    """Convenience: load YAML config and construct :class:`CitationGraphManager`."""
    cfg = load_app_config(Path(config_path))
    return CitationGraphManager(cfg=cfg)
