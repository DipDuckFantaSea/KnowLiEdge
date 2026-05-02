from __future__ import annotations

"""Autonomous literature-driven review: multi-query fused (FTS+vec+RRF) search → evidence aggregation → zh report."""

import re
import sqlite3
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from knotliedge.config.load import load_app_config
from knotliedge.storage.chroma_store import ChromaStore
from knotliedge.storage.fts_store import default_fts_db_path

from scripts.run_local_fused_search import run_fused_local_search


@dataclass
class ChunkEvidence:
    chunk_id: str
    best_rrf: float = 0.0
    queries: Set[str] = field(default_factory=set)
    doc_id: str = ""
    short_name: str = ""
    section: Optional[str] = None
    preview: str = ""
    source_md: str = ""
    fts_rank: Optional[int] = None
    vec_rank: Optional[int] = None


def _load_doc_meta(cfg, doc_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    db = default_fts_db_path(cfg)
    out: Dict[str, Dict[str, Any]] = {}
    if not db.exists():
        return out
    con = sqlite3.connect(str(db))
    cur = con.cursor()
    for did in doc_ids:
        if not did:
            continue
        row = cur.execute(
            "select doc_id, openalex_title, doi, publication_year, journal_name, citation_count "
            "from documents where doc_id = ?",
            (did,),
        ).fetchone()
        if row:
            out[did] = {
                "doc_id": row[0],
                "title": row[1] or "",
                "doi": row[2] or "",
                "year": row[3],
                "journal": row[4] or "",
                "cites": row[4] if False else row[5],
            }
    con.close()
    return out


def _snippet(s: str, *, width: int = 420) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip())
    return textwrap.shorten(t, width=width, placeholder="…")


def _collect(cfg, config_path: Path, queries: List[str]) -> Dict[str, ChunkEvidence]:
    pool: Dict[str, ChunkEvidence] = {}
    for q in queries:
        payload = run_fused_local_search(
            cfg=cfg,
            config_path=config_path,
            query=q,
            top_k=18,
            top_k_fts=45,
            top_k_vec=45,
            rrf_k=60,
            doc_id=None,
            short_name=None,
            fts_mode="bm25",
        )
        for r in payload.get("results") or []:
            cid = str(r.get("chunk_id") or "")
            if not cid:
                continue
            sc = float(r.get("rrf_score") or 0.0)
            ev = pool.get(cid)
            if ev is None:
                pool[cid] = ChunkEvidence(
                    chunk_id=cid,
                    best_rrf=sc,
                    queries={q},
                    doc_id=str(r.get("doc_id") or ""),
                    short_name=str(r.get("short_name") or ""),
                    section=r.get("section"),
                    preview=str(r.get("preview") or ""),
                    source_md=str(r.get("source_md") or ""),
                    fts_rank=r.get("fts_rank"),
                    vec_rank=r.get("vec_rank"),
                )
            else:
                ev.queries.add(q)
                if sc > ev.best_rrf:
                    ev.best_rrf = sc
                    # refresh locator fields from stronger hit
                    ev.doc_id = str(r.get("doc_id") or ev.doc_id)
                    ev.short_name = str(r.get("short_name") or ev.short_name)
                    ev.section = r.get("section") or ev.section
                    ev.preview = str(r.get("preview") or ev.preview)
                    ev.source_md = str(r.get("source_md") or ev.source_md)
                    ev.fts_rank = r.get("fts_rank")
                    ev.vec_rank = r.get("vec_rank")
    return pool


def _diverse_top(pool: Dict[str, ChunkEvidence], *, top_n: int, per_doc: int) -> List[ChunkEvidence]:
    """Pick top chunks by RRF while capping how many chunks per doc_id (reduce single-paper dominance)."""
    ranked_all = sorted(pool.values(), key=lambda x: x.best_rrf, reverse=True)
    picked: List[ChunkEvidence] = []
    per: Dict[str, int] = defaultdict(int)
    for ev in ranked_all:
        if len(picked) >= int(top_n):
            break
        d = ev.doc_id or "_"
        if per[d] >= int(per_doc):
            continue
        per[d] += 1
        picked.append(ev)
    return picked


def _write_report(
    *,
    out_path: Path,
    cfg,
    store: ChromaStore,
    pool: Dict[str, ChunkEvidence],
    top_n: int,
    per_doc_cap: int,
    expand_window: int,
    excerpt_chars: int,
) -> None:
    ranked = _diverse_top(pool, top_n=int(top_n), per_doc=int(per_doc_cap))
    doc_ids = sorted({e.doc_id for e in ranked if e.doc_id})
    meta = _load_doc_meta(cfg, doc_ids)

    lines: List[str] = []
    lines.append("# 硅基氮化镓（GaN-on-Si）器件建模综述与完整建模流程（本地知识库驱动）")
    lines.append("")
    lines.append("> 本文档由 KnotLiEdge 本地索引自动生成：多轮 **FTS5 BM25 + Chroma 向量检索 + RRF 融合** 召回证据，再对高分 `chunk_id` 做 **Chroma 邻域展开** 并人工结构化写作。证据锚点以 `chunk_id` 标注，便于你在 MCP 中用 `get_knowledge_chunk` 复核。")
    lines.append("")
    lines.append("## 0. 检索与写作方法")
    lines.append("")
    lines.append(
        "- **知识库**：`sandbox` 配置下的 Markdown vault + FTS（`04_fts_db/fts.sqlite3`）+ Chroma 向量库。\n"
        "- **融合排序**：各路查询结果在 `chunk_id` 维度去重后，以 **最大 RRF 分数**作为该分块的全局强度；正文引用使用该分数排序后的 Top 证据。\n"
        "- **连字符处理**：BM25 查询中形如 `GaN-on-Si` 的技术词在索引侧已做规范化，避免 FTS5 将 `-` 解析为 NOT 运算符导致检索失败。"
    )
    lines.append("")
    lines.append("### 0.1 本轮使用的主题查询（多轮）")
    lines.append("")
    for i, q in enumerate(SEARCH_QUERIES, start=1):
        lines.append(f"{i}. `{q}`")
    lines.append("")
    lines.append("### 0.2 证据去重与多样性策略")
    lines.append("")
    lines.append(
        f"- 先在 **{len(pool)}** 个唯一 `chunk_id` 上聚合多查询 RRF 峰值分数；\n"
        f"- 最终正文证据表取 Top **{top_n}**，并对同一 `doc_id` **最多保留 {per_doc_cap} 条**分块，避免单篇文献在摘录区过度挤占版面。"
    )
    lines.append("")

    lines.append("## 1. 执行摘要")
    lines.append("")
    lines.append(
        "硅基氮化镓（GaN-on-Si）在 **成本** 与 **同硅 CMOS 单片集成潜力** 上相对 GaN-on-SiC 有明显优势，但建模必须显式处理 **Si 衬底相关的射频漏电/损耗**、**缓冲层与体陷阱导致的色散与动态导通电阻退化**、以及 **自热与温度依赖** 等耦合效应。工程上通常采用 **“物理/TCAD 机理 → 等效电路或紧凑模型 → 电热与封装寄生 → 电路/系统验证”** 的分层流程；在平台化阶段再与 **PDK / DTCO** 流程对齐。"
    )
    lines.append("")

    lines.append("## 2. GaN-on-Si 建模动机：与 SiC 衬底路线的结构差异")
    lines.append("")
    lines.append(
        "GaN-on-Si 的核心矛盾是：**在低成本 Si 上生长高质量 GaN 外延** 的同时，**衬底电学行为**（漏电、位移电流、损耗）与 **热学行为**（导热差、自热）会强烈反馈到器件端特性。文献中常见表述包括：为降低成本采用 GaN-on-Si；但 RF 漏电与高温漏电会进一步恶化自热与效率（见下方证据摘录）。"
    )
    lines.append("")

    lines.append("## 3. 建模分层体系（建议作为团队分工边界）")
    lines.append("")
    lines.append("| 层级 | 主要对象 | 典型产出 | 与 GaN-on-Si 的强相关点 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| L0 材料与外延 | 缓冲层、沟道、势垒、掺杂 | 材料参数表、缺陷模型假设 | 缓冲陷阱、C/Fe 掺杂、漏电路径 |")
    lines.append("| L1 器件物理 / TCAD | 2DEG、极化、场板、栅栈 | 结构敏感曲线、机理图 | 衬底漏电、反型层电子、TCAD 验证 |")
    lines.append("| L1' 射频寄生与无源 | 螺旋电感、变压器、BEOL | 集总/EM 等效、scalable model | RF GaN-on-Si 无源建模缺口 |")
    lines.append("| L2 紧凑模型 / 等效电路 | Angelov、ASM、子电路 | Verilog-A / SPICE card | 衬底网络、色散、NQS |")
    lines.append("| L3 电热耦合 | 自热、热阻网络 | 温度依赖参数、脉冲测试策略 | 与动态测试条件强绑定 |")
    lines.append("| L4 封装与系统 | 驱动、并联、EMI、散热 | 系统级仿真、布局规则 | 功率模块与热路径 |")
    lines.append("| L5 平台 / PDK / DTCO | 规则、角模型、良率 | PDK、DTCO KPI | 与 CMOS 集成、工艺窗口 |")
    lines.append("")

    lines.append("## 4. 主题综述（按证据组织）")
    lines.append("")

    lines.append("### 4.1 衬底射频漏电与温度：从 TCAD 机理到准物理等效电路")
    lines.append("")
    lines.append(
        "该方向强调 **Si 衬底并非理想绝缘支撑体**：在高压/高温/射频条件下，缓冲层—Si 界面附近的载流子行为可主导 **RF leakage**，并与 **自热** 形成正反馈。文献路线通常是 **TCAD 揭示机理**（载流子浓度、反型层电子等）→ **在经典大信号模型（如 Angelov-GaN）上叠加 C–R–C 衬底网络** → 用功率扫描/温度扫描验证。"
    )
    lines.append("")

    lines.append("### 4.2 动态效应、陷阱与“电流塌陷 / 膝点外推”")
    lines.append("")
    lines.append(
        "GaN 功率与射频模型往往需要同时解释 **静态 I–V** 与 **脉冲/动态** 特性差异：色散机制可导致 **knee walkout** 与 **current collapse** 等现象。工程上常用 **多偏置 S 参数 / 脉冲 I–V** 提取色散参数，并在紧凑模型或子电路里用 **RC 弛豫支路** 近似陷阱动力学（具体网络形式依模型族而异）。"
    )
    lines.append("")

    lines.append("### 4.3 面向 RF 功放与宽带应用的大信号建模")
    lines.append("")
    lines.append(
        "在 **GaN-on-Si 衬底** 上设计功率放大器时，除本征晶体管外，还需处理 **非线性电荷—电流关系**、**栅漏电** 与 **衬底/无源寄生** 的宽带效应。文献中可见“脉冲与静态测量联合建模”的路线，用于在电路仿真中复现色散相关行为。"
    )
    lines.append("")

    lines.append("### 4.4 无源器件与 BEOL：RF GaN-on-Si 的“建模空白”")
    lines.append("")
    lines.append(
        "当系统走向 **MMIC / 异质集成** 时，**螺旋电感/变压器** 等无源结构在 GaN-on-Si 上的 **可扩展集总模型** 与 EM 堆叠描述成为瓶颈之一；文献指出相较硅 RF 无源，GaN-on-Si 上公开模型与数据明显不足，需要 **scalable lumped model** 或 **EM stack-up** 支撑定制电感设计。"
    )
    lines.append("")

    lines.append("### 4.5 高温、DTCO 与 CAD 框架：从器件表征到规模化")
    lines.append("")
    lines.append(
        "当目标扩展到 **高温环境** 或 **DTCO** 时，需要把 **器件表征、建模、电路仿真与版图/工艺 KPI** 串成可回归的闭环；文献中可见 **HT GaN-on-Si** 与 **实验验证的 CAD 框架** 的路线，用于支撑后续规模化与 mixed-signal 场景。"
    )
    lines.append("")

    lines.append("### 4.6 与硅 CMOS 集成与 PDK 形态")
    lines.append("")
    lines.append(
        "在 **300 mm GaN-on-Si(111) + 集成 Si CMOS** 路线中，文献会明确提到 **PDK in-development**：包含晶体管/互连规则、紧凑模型、Pcell 与 tape-in 流程等。这意味着建模交付物不仅是“单管模型”，而是 **可 tape-out 的规则+模型+验证用例** 的组合。"
    )
    lines.append("")

    lines.append("## 5. 一套可落地的完整建模流程（建议作为 SOP）")
    lines.append("")
    lines.append("### 5.1 流程总览（Mermaid）")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TB")
    lines.append("  A[目标定义: 功率/射频/温度/电压等级] --> B[测试规划: DC/脉冲/S/负载牵引/热成像]")
    lines.append("  B --> C[外延与结构参数收集: 厚度/掺杂/场板/缓冲]")
    lines.append("  C --> D{是否需要机理解释?}")
    lines.append("  D -- 是 --> E[TCAD: 极化/2DEG/缓冲/衬底漏电/自热]")
    lines.append("  D -- 否 --> F[直接紧凑建模/等效电路]")
    lines.append("  E --> G[机理敏感项 → 等效网络/紧凑模型参数]")
    lines.append("  F --> G")
    lines.append("  G --> H[电热: 脉冲条件/热阻/温度依赖参数]")
    lines.append("  H --> I[无源/寄生: BEOL/衬底/焊盘去嵌]")
    lines.append("  I --> J[电路级验证: PA/电源/驱动/EMI]")
    lines.append("  J --> K{是否满足 KPI?}")
    lines.append("  K -- 否 --> B")
    lines.append("  K -- 是 --> L[PDK化: 角模型/规则/文档/QA]")
    lines.append("```")
    lines.append("")

    lines.append("### 5.2 分阶段输入 / 输出（I/O）清单")
    lines.append("")
    lines.append("| 阶段 | 关键输入 | 关键输出 | 质量门禁（示例） |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| P0 需求冻结 | 拓扑、电压、频率、占空比、散热边界 | 需求矩阵、测试矩阵 | 覆盖自热/关断应力 |")
    lines.append("| P1 数据获取 | wafer 级 CV/IV/脉冲/S | 清洗后数据集、去嵌报告 | 可重复性/探针一致性 |")
    lines.append("| P2 TCAD（可选） | 结构剖面、材料参数 | 场/载流子/温度分布 | 与关键电学趋势一致 |")
    lines.append("| P3 核心模型 | P1+P2 | 可仿真 deck、参数表 | 多偏置拟合残差阈值 |")
    lines.append("| P4 电热 | 热阻、脉冲条件 | T-dependent 参数 | 温度扫描误差阈值 |")
    lines.append("| P5 系统 | 封装/驱动/母线寄生 | 系统仿真网表 | EMI/效率/热点的联合达标 |")
    lines.append("| P6 PDK | 规则/角/统计 | PDK bundle | tape-in checklist |")
    lines.append("")

    lines.append("## 6. 关键证据摘录（Top 分块，含 chunk_id）")
    lines.append("")
    for i, ev in enumerate(ranked, start=1):
        m = meta.get(ev.doc_id, {})
        title = m.get("title") or ev.short_name or ev.doc_id
        lines.append(f"### 6.{i} `chunk_id={ev.chunk_id}`  RRF≈{ev.best_rrf:.5f}")
        lines.append("")
        lines.append(f"- **题名**: {title}")
        if m.get("doi"):
            lines.append(f"- **DOI**: `{m['doi']}`")
        if m.get("year"):
            lines.append(f"- **年份**: {m['year']}")
        lines.append(f"- **章节**: {ev.section}")
        lines.append(f"- **命中查询数**: {len(ev.queries)}")
        lines.append("")
        lines.append("**检索预览**:")
        lines.append("")
        lines.append("```")
        lines.append(_snippet(ev.preview, width=900))
        lines.append("```")
        lines.append("")
        try:
            ctx = store.get_context_by_chunk_id(ev.chunk_id, window=int(expand_window))
            body = (ctx.get("text") or "").strip()
            cap = int(excerpt_chars)
            if len(body) > cap:
                body = body[:cap] + "\n…（截断）"
            lines.append("**邻域展开（倒查原文）**:")
            lines.append("")
            lines.append("```")
            lines.append(body)
            lines.append("```")
        except Exception as e:
            lines.append(f"_展开失败：{e}_")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## 7. 参考文献表（按 doc_id，OpenAlex 元数据）")
    lines.append("")
    lines.append("| doc_id | 年份 | 引用数 | DOI | 题名 |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for did in doc_ids:
        m = meta.get(did, {})
        cites = m.get("cites")
        if cites is None:
            cites = ""
        lines.append(
            f"| `{did}` | {m.get('year') or ''} | {cites} | {m.get('doi') or ''} | {m.get('title') or ''} |"
        )
    lines.append("")

    lines.append("## 8. 结论与后续工作建议")
    lines.append("")
    lines.append(
        "1. **先把 GaN-on-Si 的“衬底—缓冲—沟道”漏电与色散路径写清楚**，再决定 TCAD 与等效网络的边界；否则紧凑模型会在高压/宽带条件下系统性失效。\n"
        "2. **测试计划必须覆盖脉冲与温度**，并与自热策略（占空比、封装热阻）绑定；仅静态 I–V 很难约束动态参数。\n"
        "3. **无源与 BEOL** 在 RF GaN-on-Si 上经常是第二瓶颈：需要与晶体管模型同等级别的 QA。\n"
        "4. 若目标是量产/异质集成，应尽早按 **PDK 交付物** 定义模型接口（角、规则、验证用例），避免“实验室精确但不可 tape-in”的模型。"
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


SEARCH_QUERIES: List[str] = [
    "硅基氮化镓 GaN on Si HEMT 器件建模 综述 2DEG 缓冲层",
    "GaN on Si RF leakage substrate TCAD equivalent circuit temperature",
    "GaN on Si current collapse knee walkout dispersion trapping self heating",
    "GaN HEMT on Si compact model SPICE Verilog A Angelov large signal",
    "GaN on Si power transistor p GaN E mode MIS HEMT dynamic RDSon buffer traps",
    "GaN on Si mmWave RF integrated CMOS PDK compact model BEOL",
    "GaN on Si high temperature DTCO CAD framework circuit simulation",
    "GaN on Si spiral inductor scalable lumped model electromagnetic BEOL",
]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="sandbox/configs/sandbox.yaml")
    p.add_argument("--out", type=str, default="output/reports/gan_on_si_modeling_autonomous_review_zh.md")
    p.add_argument("--top", type=int, default=16)
    p.add_argument("--per-doc", type=int, default=2, help="Max chunks per doc_id in final evidence table.")
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--excerpt", type=int, default=3200)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_app_config(config_path)
    store = ChromaStore(cfg=cfg, embedder=None)

    pool = _collect(cfg, config_path, SEARCH_QUERIES)
    out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    _write_report(
        out_path=out_path,
        cfg=cfg,
        store=store,
        pool=pool,
        top_n=int(args.top),
        per_doc_cap=int(args.per_doc),
        expand_window=int(args.window),
        excerpt_chars=int(args.excerpt),
    )
    print(str(out_path))


if __name__ == "__main__":
    main()
