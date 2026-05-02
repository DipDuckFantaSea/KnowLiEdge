"""Run ``search_hybrid_knowledge_and_radar`` logic offline and write Markdown output.

Requires embedding IPC / Chroma / FTS / venue radar data same as MCP runtime.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Optional

from knotliedge.config.load import load_app_config
from knotliedge.embeddings.lazy_ipc_gate import LazyIpcEmbedderSession
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.mcp.hybrid_knowledge_radar_search import run_hybrid_knowledge_and_radar_search
from knotliedge.storage.chroma_store import ChromaStore, format_chroma_http_client_error
from knotliedge.storage.venue_radar_store import default_venue_radar_db_path


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Run hybrid local + venue radar search; write Markdown report.")
    parser.add_argument("--config", type=str, default="sandbox/configs/sandbox.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--query",
        type=str,
        default="GaN HEMT flip-chip power amplifier microwave",
        help="Search query text.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output Markdown path (default: beside venue_radar.sqlite3 as _last_hybrid_search.md).",
    )
    parser.add_argument("--top-k-total", type=int, default=20)
    parser.add_argument("--top-k-local-docs", type=int, default=10)
    parser.add_argument("--top-k-radar-works", type=int, default=10)
    parser.add_argument("--local-chunks-per-doc", type=int, default=3)
    parser.set_defaults(ai_parse_query=True)
    parser.add_argument(
        "--no-ai-parse-query",
        dest="ai_parse_query",
        action="store_false",
        help="Disable LLM keyword extraction before local fusion (heuristic split only).",
    )
    parser.add_argument(
        "--ai-parse-query",
        dest="ai_parse_query",
        action="store_true",
        help="Enable LLM keyword extraction (default). DeepSeek Chat Completions: DEEPSEEK_API_KEY or OPENAI_API_KEY; template templates/openai_chat/local_research_keyword_en_terms.json (deepseek-v4-flash).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_app_config(config_path)
    embed_gate = LazyIpcEmbedderSession(config_path=config_path)
    try:
        store = ChromaStore(cfg=cfg, embedder=None)
    except Exception as e:
        hint = str(e)
        if "无法连接 Chroma HTTP" not in hint and "已连上服务端" not in hint:
            hint = format_chroma_http_client_error(cfg, e)
        logger.error("ChromaStore 初始化失败（HTTP 连接/集合打开）\n%s", hint)
        logger.debug("%s", traceback.format_exc())
        raise SystemExit(2) from e

    out_path = Path(args.out).resolve() if args.out else (default_venue_radar_db_path(cfg).parent / "_last_hybrid_search.md")

    def embed_q(q: str):
        last_err: Optional[BaseException] = None
        for attempt in range(2):
            try:
                embed_gate.require_for_store(store)
                emb = embed_gate.get_optional()
                if emb is None:
                    raise RuntimeError("Embedder is not available after require_for_store().")
                return getattr(emb, "embed_query")(q)
            except Exception as e:
                last_err = e
                if attempt == 0:
                    logger.warning("embed_query failed; invalidate session and retry once | %s", e)
                    embed_gate.invalidate()
                else:
                    break
        raise RuntimeError(f"embed_query failed after retry: {last_err}") from last_err

    try:
        out = run_hybrid_knowledge_and_radar_search(
            cfg,
            store,
            embed_q,
            args.query,
            top_k_total=int(args.top_k_total),
            top_k_local_docs=int(args.top_k_local_docs),
            top_k_radar_works=int(args.top_k_radar_works),
            local_chunks_per_doc=int(args.local_chunks_per_doc),
            report_markdown_dir=out_path.parent,
            ai_parse_query=bool(args.ai_parse_query),
        )
    except Exception as e:
        logger.error("混合检索失败: %s", e)
        logger.error(
            "若与 Chroma 通信失败，请确认 HTTP 端点 %s:%s 上已有守护进程，且 persist 目录为 %s",
            cfg.chroma.http_host,
            int(cfg.chroma.http_port),
            cfg.paths.chroma_db_dir,
        )
        logger.debug("%s", traceback.format_exc())
        raise SystemExit(3) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(str(out.get("markdown") or ""), encoding="utf-8")
    logger.info("Wrote hybrid search markdown | path=%s", out_path)


if __name__ == "__main__":
    main()
