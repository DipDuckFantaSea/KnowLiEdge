from __future__ import annotations

"""Run MCP-equivalent fused local search, expand each hit via ChromaStore context, write a Chinese report."""

import argparse
import sqlite3
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from knotliedge.config.load import load_app_config
from knotliedge.storage.chroma_store import ChromaStore
from knotliedge.storage.fts_store import default_fts_db_path

from scripts.run_local_fused_search import run_fused_local_search


def _doc_titles(cfg, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    db = default_fts_db_path(cfg)
    if not db.exists():
        return {}
    con = sqlite3.connect(str(db))
    cur = con.cursor()
    out: Dict[str, Dict[str, Any]] = {}
    for did in doc_ids:
        if not did:
            continue
        row = cur.execute(
            "select doc_id, openalex_title, doi, publication_year, journal_name from documents where doc_id = ?",
            (did,),
        ).fetchone()
        if row:
            out[did] = {
                "doc_id": row[0],
                "title": row[1] or "",
                "doi": row[2] or "",
                "year": row[3],
                "journal": row[4] or "",
            }
    con.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fused RRF search + expanded context → Markdown report (zh).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--expand-window", type=int, default=2)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    config_path = Path(args.config)
    cfg = load_app_config(config_path)

    fused = run_fused_local_search(
        cfg=cfg,
        config_path=config_path,
        query=args.query,
        top_k=int(args.top_k),
        top_k_fts=50,
        top_k_vec=50,
        rrf_k=60,
        doc_id=None,
        short_name=None,
        fts_mode="bm25",
    )

    store = ChromaStore(cfg=cfg, embedder=None)
    doc_ids = sorted({str(r.get("doc_id") or "") for r in fused.get("results") or [] if r.get("doc_id")})
    titles = _doc_titles(cfg, doc_ids)

    lines: List[str] = []
    lines.append("# 融合检索（FTS + 向量 + RRF）调研报告")
    lines.append("")
    lines.append(f"- **配置**: `{config_path}`")
    lines.append(f"- **查询**: {fused.get('query')}")
    lines.append(f"- **RRF k**: {fused.get('rrf_k')}，**top_k**: {fused.get('top_k')}")
    lines.append(f"- **FTS 模式**: {fused.get('fts_mode')}，**keyphrase**: `{fused.get('keyphrase')}`")
    lines.append("")
    lines.append("## 方法说明")
    lines.append("")
    lines.append(
        "本报告使用与 MCP 工具 `universal_academic_search`（`mode=local` 且 `fused=True`）等价的流程："
        "并行执行 FTS5（BM25）与 Chroma 向量检索，再用 `rrf_merge` 融合排序；"
        "对每个命中的 `chunk_id` 调用 `ChromaStore.get_context_by_chunk_id(..., window=...)` 展开相邻分块作为“倒查原文”上下文。"
    )
    lines.append("")

    for i, r in enumerate(fused.get("results") or [], start=1):
        cid = str(r.get("chunk_id") or "")
        did = str(r.get("doc_id") or "")
        meta = titles.get(did, {})
        title = meta.get("title") or r.get("short_name") or did
        lines.append(f"## 命中 #{i}（RRF={r.get('rrf_score'):.6f}）")
        lines.append("")
        lines.append(f"- **chunk_id**: `{cid}`")
        lines.append(f"- **doc_id**: `{did}`")
        lines.append(f"- **题名（OpenAlex/documents）**: {title}")
        if meta.get("doi"):
            lines.append(f"- **DOI**: `{meta['doi']}`")
        if meta.get("year"):
            lines.append(f"- **年份**: {meta['year']}")
        if meta.get("journal"):
            lines.append(f"- **期刊/来源字段**: {meta['journal']}")
        lines.append(f"- **章节（chunk metadata）**: {r.get('section')}")
        lines.append(f"- **FTS 排名 / 分数**: rank={r.get('fts_rank')} score={r.get('fts_score')}")
        lines.append(f"- **向量排名 / 距离**: rank={r.get('vec_rank')} score={r.get('vec_score')}")
        lines.append(f"- **source_md**: `{r.get('source_md')}`")
        lines.append("")
        lines.append("### 检索预览（FTS 分支优先）")
        lines.append("")
        prev = (r.get("preview") or "").strip()
        if prev:
            lines.append("```")
            lines.append(textwrap.shorten(prev.replace("```", "`"), width=1200, placeholder="…"))
            lines.append("```")
        else:
            lines.append("_（无 preview）_")
        lines.append("")
        lines.append(f"### 倒查原文（Chroma 展开 window={int(args.expand_window)}）")
        lines.append("")
        try:
            ctx = store.get_context_by_chunk_id(cid, window=int(args.expand_window))
            body = (ctx.get("text") or "").strip()
            lines.append("```")
            lines.append(body[:12000] + ("…\n（截断至 12000 字符）" if len(body) > 12000 else ""))
            lines.append("```")
        except Exception as e:
            lines.append(f"_展开失败：{e}_")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(out_path.resolve()))


if __name__ == "__main__":
    main()
