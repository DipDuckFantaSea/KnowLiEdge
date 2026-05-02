from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from knotliedge.chunking.md_chunker import chunk_markdown, load_markdown_doc
from knotliedge.config.types import AppConfig
from knotliedge.embeddings import get_embedder
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.mineru.http_client import (
    get_task_result,
    get_task_result_bytes,
    get_task_status,
    parse_zip_result,
    submit_task,
)
from knotliedge.mineru.service import start_service, stop_service
from knotliedge.ops.runtime_paths import get_runtime_paths
from knotliedge.pipeline.doc_id import compute_doc_id
from knotliedge.pipeline.frontmatter import Frontmatter, now_iso8601, write_markdown_with_frontmatter
from knotliedge.pipeline.metadata_extract import extract_paper_metadata
from knotliedge.pipeline.short_name import generate_short_name
from knotliedge.storage.chroma_store import ChromaStore
from knotliedge.storage.schema import ChunkMetadata, now_iso8601 as now_iso8601_chunks

logger = setup_logging()


@dataclass(frozen=True)
class WatchStats:
    seen_total: int
    processed_ok: int
    processed_skipped: int
    processed_failed: int
    parse_seconds_ok: List[float]


def _iter_pdfs(raw_pdf_dir: Path) -> Iterable[Path]:
    # Recursive + case-insensitive extensions
    yield from sorted(set(raw_pdf_dir.rglob("*.pdf")) | set(raw_pdf_dir.rglob("*.PDF")))


def _state_path(cfg: AppConfig) -> Path:
    return cfg.project_root / ".knotliedge" / "watch_state.jsonl"


def _load_seen_doc_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen: Set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                doc_id = str(obj.get("doc_id") or "")
                status = str(obj.get("status") or "")
                if doc_id and status in {"ok", "skipped"}:
                    seen.add(doc_id)
            except Exception:
                continue
    except Exception:
        return set()
    return seen


def _append_state(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    path.write_text(path.read_text(encoding="utf-8", errors="ignore") + line + "\n", encoding="utf-8") if path.exists() else path.write_text(line + "\n", encoding="utf-8")


def _extract_markdown_from_mineru_result(result: Dict[str, Any]) -> str:
    # MinerU result payloads may vary; try common keys.
    for k in ("markdown", "markdown_text", "md", "md_content", "content_md"):
        v = result.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # Some responses wrap outputs under a list: {"results": [{...}]}
    results = result.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            return _extract_markdown_from_mineru_result(first)
    # MinerU 3.x pipeline may return {"results": {<doc_id>: {"md_content": "..."} } }
    if isinstance(results, dict) and results:
        for _doc_id, payload in results.items():
            if isinstance(payload, dict):
                return _extract_markdown_from_mineru_result(payload)
    # Some responses wrap in {"result": {...}}
    inner = result.get("result")
    if isinstance(inner, dict):
        return _extract_markdown_from_mineru_result(inner)
    raise RuntimeError(f"MinerU result does not contain markdown text keys. keys={list(result.keys())[:50]}")


def _copy_to_temp_pdf(cfg: AppConfig, *, src_pdf: Path, doc_id: str) -> Path:
    """Copy PDF to an ASCII-safe temp path to avoid backend issues with unicode/long filenames."""
    tmp_dir = cfg.project_root / ".knotliedge" / "tmp_pdfs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_pdf = tmp_dir / f"{doc_id}.pdf"
    if not tmp_pdf.exists() or tmp_pdf.stat().st_size != src_pdf.stat().st_size:
        shutil.copy2(src_pdf, tmp_pdf)
    return tmp_pdf


def _parse_pdf_via_mineru_api(
    *,
    cfg: AppConfig,
    api_url: str,
    pdf_path: Path,
    doc_id: str,
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    return_images: bool = False,
    timeout_s: int = 600,
) -> Tuple[str, Dict[str, bytes]]:
    """Submit async task and poll until result is ready, returning markdown text."""
    safe_pdf = _copy_to_temp_pdf(cfg, src_pdf=pdf_path, doc_id=doc_id)
    try:
        submitted = submit_task(
            api_url=api_url,
            pdf_paths=[safe_pdf],
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            return_md=True,
            return_images=bool(return_images),
            response_format_zip=bool(return_images),
            timeout_s=60,
        )
    except Exception as e:
        raise RuntimeError(
            f"MinerU submit_task failed: api_url={api_url} pdf={pdf_path} backend={backend} method={parse_method} | {e}"
        ) from e
    task_id = submitted.task_id
    t0 = time.time()
    while time.time() - t0 < float(timeout_s):
        try:
            st = get_task_status(api_url=api_url, task_id=task_id, timeout_s=20)
        except Exception as e:
            raise RuntimeError(f"MinerU get_task_status failed: api_url={api_url} task_id={task_id} | {e}") from e
        status = str(st.get("status") or "")
        if status in {"completed", "succeeded", "success"}:
            if return_images:
                try:
                    raw, hdrs = get_task_result_bytes(api_url=api_url, task_id=task_id, timeout_s=300)
                    content_type = str(hdrs.get("content-type") or "").lower()
                    is_zip = raw[:2] == b"PK" or "zip" in content_type
                    if is_zip:
                        _md_name, md_bytes, images = parse_zip_result(zip_bytes=raw)
                        md_text = md_bytes.decode("utf-8", errors="replace") if md_bytes else ""
                        if md_text.strip():
                            return md_text, images
                except Exception as e:
                    logger.warning("MinerU zip result fetch/parse failed, falling back to json. task_id=%s err=%s", task_id, e)

            try:
                res = get_task_result(api_url=api_url, task_id=task_id, timeout_s=60)
            except Exception as e:
                raise RuntimeError(f"MinerU get_task_result failed: api_url={api_url} task_id={task_id} | {e}") from e
            try:
                return _extract_markdown_from_mineru_result(res), {}
            except Exception as e:
                dump_dir = (cfg.project_root / ".knotliedge").resolve()
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / f"last_mineru_result_{task_id}.json"
                try:
                    dump_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
                raise RuntimeError(f"MinerU result markdown extraction failed. dumped={dump_path}") from e
        if status in {"failed", "error"}:
            raise RuntimeError(f"MinerU task failed: task_id={task_id} | {st}")
        time.sleep(0.5)
    raise TimeoutError(f"MinerU task timeout: task_id={task_id} pdf={pdf_path}")


def _index_one_md(*, cfg: AppConfig, store: ChromaStore, embedder: Any, md_path: Path, purge_doc: bool) -> int:
    doc = load_markdown_doc(md_path)
    if purge_doc:
        _ = store.delete_by_doc_id(doc.doc_id)

    chunks = chunk_markdown(doc, cfg.chunking)
    if not chunks:
        return 0

    doc_hash = hashlib.sha1((doc.body or "").encode("utf-8", errors="ignore")).hexdigest()
    try:
        source_md_mtime_ns = int(doc.source_md.resolve().stat().st_mtime_ns)
    except Exception:
        source_md_mtime_ns = None

    created_at = now_iso8601_chunks()
    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []
    for c in chunks:
        ids.append(c.chunk_id)
        docs.append(c.text)
        meta = ChunkMetadata(
            doc_id=c.doc_id,
            short_name=doc.short_name,
            chunk_id=c.chunk_id,
            source_md=str(c.source_md.resolve()),
            source_md_mtime_ns=source_md_mtime_ns,
            section=c.section,
            chunk_index=c.chunk_index,
            text_len=len(c.text),
            created_at=created_at,
            doc_hash=doc_hash,
        )
        metas.append(meta.to_chroma())

    embs = embedder.embed_texts(docs)
    store.upsert_chunks(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(chunks)


def run_watch_ingest(
    cfg: AppConfig,
    *,
    config_path: Path,
    api_url: Optional[str] = None,
    enable_vlm_preload: bool = True,
    poll_interval_s: float = 3.0,
    purge_doc: bool = True,
    parse_backend: str = "pipeline",
    parse_method: str = "txt",
    formula_enable: bool = True,
    table_enable: bool = True,
    export_images: bool = False,
    max_loops: Optional[int] = None,
    max_docs: Optional[int] = None,
    parse_mode: str = "api",
    mineru_model_source: str = "local",
    mineru_virtual_vram_size: Optional[int] = 8,
) -> WatchStats:
    """Watch raw_pdf_dir and ingest new PDFs into vault + ChromaDB.

    Behavior:
    - Uses doc_id as dedupe key (stable for unchanged files).
    - For each new PDF:
      1) Parse via mineru-api (sync)
      2) Write Markdown with frontmatter into markdown_vault_dir
      3) Incrementally upsert into Chroma (purge-by-doc_id then upsert)

    Args:
        cfg: App config.
        config_path: YAML config path used for embedding service auto-start.
        api_url: MinerU api url. If None, will start a local mineru-api service.
        enable_vlm_preload: If starting a local service, enable VLM preload.
        poll_interval_s: Poll interval for directory scanning.
        purge_doc: Whether to purge existing chunks for same doc_id before upsert.
        parse_backend: MinerU backend.
        parse_method: MinerU parse method.
        formula_enable: MinerU formula parsing.
        table_enable: MinerU table parsing.
        export_images: If true, request ZIP result + extract images into vault/images.
        max_loops: If provided, stop after N loops (useful for tests).
        max_docs: If provided, stop after attempting N new docs (ok/failed).
        parse_mode: Must be "api" (enforced).
        mineru_model_source: Passed to mineru-api as env var MINERU_MODEL_SOURCE.
        mineru_virtual_vram_size: Passed to mineru-api as env var MINERU_VIRTUAL_VRAM_SIZE.

    Returns:
        WatchStats summary.
    """
    state_path = _state_path(cfg)
    seen = _load_seen_doc_ids(state_path)

    parse_mode = str(parse_mode or "api").strip().lower()
    if parse_mode != "api":
        raise ValueError(f"parse_mode 必须为 api（已强制使用 mineru-api），当前={parse_mode}")

    use_api_url = api_url
    started_local_service = False
    if not use_api_url:
        env_overrides = {
            "MINERU_MODEL_SOURCE": str(mineru_model_source),
        }
        if mineru_virtual_vram_size is not None:
            env_overrides["MINERU_VIRTUAL_VRAM_SIZE"] = str(int(mineru_virtual_vram_size))
        rp = get_runtime_paths(cfg)
        st = start_service(
            enable_vlm_preload=bool(enable_vlm_preload),
            env_overrides=env_overrides,
            log_dir=rp.mineru_logs_dir,
            work_dir=rp.mineru_work_dir,
        )
        use_api_url = st.api_url
        started_local_service = True

    try:
        embedder = get_embedder(config_path=Path(config_path))
    except Exception as e:
        raise RuntimeError(str(e)) from e
    store = ChromaStore(cfg=cfg, embedder=embedder)  # type: ignore[arg-type]

    seen_total = 0
    ok = 0
    skipped = 0
    failed = 0
    parse_seconds_ok: List[float] = []
    attempted = 0

    loops = 0
    logger.info("Watch ingest started. raw_pdf_dir=%s api_url=%s", cfg.paths.raw_pdf_dir, use_api_url)
    try:
        while True:
            loops += 1

            pdf_iter: Iterable[Path]
            if max_docs is not None:
                # For small-sample runs, prefer smaller PDFs first to get quick successful timings.
                all_pdfs = list(_iter_pdfs(cfg.paths.raw_pdf_dir))
                all_pdfs.sort(key=lambda p: (p.stat().st_size if p.exists() else 0, str(p)))
                pdf_iter = all_pdfs
            else:
                pdf_iter = _iter_pdfs(cfg.paths.raw_pdf_dir)

            for pdf_path in pdf_iter:
                seen_total += 1
                try:
                    doc_id = compute_doc_id(pdf_path)
                    if doc_id in seen:
                        continue

                    md_name = cfg.markdown.vault_filename_pattern.format(doc_id=doc_id)
                    md_path = cfg.paths.markdown_vault_dir / md_name
                    if md_path.exists():
                        seen.add(doc_id)
                        skipped += 1
                        _append_state(
                            state_path,
                            {
                                "ts": now_iso8601(),
                                "status": "skipped",
                                "doc_id": doc_id,
                                "pdf_path": str(pdf_path.resolve()),
                                "reason": "md_exists",
                            },
                        )
                        continue

                    if not use_api_url:
                        raise RuntimeError("MinerU api_url is not set.")

                    t0 = time.perf_counter()
                    t_parse0 = time.perf_counter()
                    try:
                        markdown_text, images = _parse_pdf_via_mineru_api(
                            cfg=cfg,
                            api_url=use_api_url,
                            pdf_path=pdf_path,
                            doc_id=doc_id,
                            backend=parse_backend,
                            parse_method=parse_method,
                            formula_enable=formula_enable,
                            table_enable=table_enable,
                            return_images=bool(export_images),
                        )
                    finally:
                        parse_seconds = time.perf_counter() - t_parse0

                    short_name, _parts = generate_short_name(
                        markdown_text=markdown_text,
                        venue=None,
                        pdf_path=pdf_path,
                    )
                    meta = extract_paper_metadata(markdown_text)
                    fm = Frontmatter(
                        doc_id=doc_id,
                        short_name=short_name,
                        source_pdf=str(pdf_path.resolve()),
                        title=meta.title or pdf_path.stem,
                        authors=meta.authors,
                        year=meta.year,
                        venue=meta.venue,
                        parsed_at=now_iso8601(),
                        parser="mineru-api",
                        version=cfg.markdown.frontmatter_version,
                    )
                    write_markdown_with_frontmatter(output_path=md_path, frontmatter=fm, markdown_body=markdown_text)

                    if images:
                        images_dir = md_path.parent / "images"
                        images_dir.mkdir(parents=True, exist_ok=True)
                        wrote = 0
                        for rel, b in images.items():
                            rel_norm = str(rel).replace("\\", "/")
                            name = Path(rel_norm).name
                            if not name:
                                continue
                            out = images_dir / name
                            try:
                                if not out.exists() or out.stat().st_size == 0:
                                    out.write_bytes(b)
                                wrote += 1
                            except Exception as e:
                                logger.warning("Write image failed: %s | %s", out, e)
                        if wrote:
                            logger.info("Exported images: %s -> %s (%s files)", doc_id, images_dir, wrote)

                    chunks_n = _index_one_md(
                        cfg=cfg,
                        store=store,
                        embedder=embedder,
                        md_path=md_path,
                        purge_doc=purge_doc,
                    )
                    dt = time.perf_counter() - t0

                    ok += 1
                    attempted += 1
                    seen.add(doc_id)
                    parse_seconds_ok.append(float(parse_seconds))
                    _append_state(
                        state_path,
                        {
                            "ts": now_iso8601(),
                            "status": "ok",
                            "doc_id": doc_id,
                            "pdf_path": str(pdf_path.resolve()),
                            "md_path": str(md_path.resolve()),
                            "chunks": int(chunks_n),
                            "parse_seconds": float(round(float(parse_seconds), 3)),
                            "seconds": float(round(dt, 3)),
                        },
                    )
                    logger.info(
                        "Ingest ok: %s | chunks=%s | parse=%.2fs | total=%.2fs",
                        pdf_path.name,
                        chunks_n,
                        float(parse_seconds),
                        dt,
                    )
                except Exception as e:
                    failed += 1
                    attempted += 1
                    _append_state(
                        state_path,
                        {
                            "ts": now_iso8601(),
                            "status": "failed",
                            "pdf_path": str(pdf_path.resolve()),
                            "error": str(e),
                        },
                    )
                    logger.error("Ingest failed: %s | %s", pdf_path, e)
                finally:
                    if max_docs is not None and attempted >= int(max_docs):
                        break

            if max_docs is not None and attempted >= int(max_docs):
                break
            if max_loops is not None and loops >= int(max_loops):
                break
            time.sleep(float(poll_interval_s))
    finally:
        # Avoid leaving mineru-api processes behind when we started one.
        if started_local_service:
            _ = stop_service()

    return WatchStats(
        seen_total=seen_total,
        processed_ok=ok,
        processed_skipped=skipped,
        processed_failed=failed,
        parse_seconds_ok=parse_seconds_ok,
    )

