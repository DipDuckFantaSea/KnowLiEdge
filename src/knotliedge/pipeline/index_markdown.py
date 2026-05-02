from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from knotliedge.chunking.md_chunker import chunk_markdown, load_markdown_doc
from knotliedge.config.types import AppConfig
from knotliedge.embeddings import get_embedder
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.metadata.document_profile import build_document_profile
from knotliedge.storage.chroma_store import ChromaStore
from knotliedge.storage.fts_store import FtsStore, default_fts_db_path
from knotliedge.storage.schema import ChunkMetadata, now_iso8601

logger = setup_logging()


def _state_path(cfg: AppConfig) -> Path:
    return cfg.project_root / ".knotliedge" / "index_markdown_state.jsonl"


def _append_state(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class IndexStats:
    total_docs: int
    total_chunks: int
    succeeded_docs: int
    failed_docs: int
    skipped_docs: int


def _iter_markdowns(vault_dir: Path) -> List[Path]:
    # Vault is typically nested (e.g. topic folders); index recursively.
    md_files = sorted(vault_dir.rglob("*.md"))
    # MinerU commonly writes extracted assets under `assets/`; never index those as primary documents.
    assets_dir = (vault_dir / "assets").resolve()
    if assets_dir.exists():
        md_files = [p for p in md_files if assets_dir not in p.resolve().parents]
    return md_files


def run_index_markdown(
    cfg: AppConfig,
    *,
    config_path: Path,
    mode: Optional[str] = None,
    limit: Optional[int] = None,
    rebuild: bool = False,
    rebuild_hard: bool = False,
    purge_doc: bool = False,
    enable_fts: bool = False,
    fts_db_path: Optional[Path] = None,
) -> IndexStats:
    """Index markdown vault into ChromaDB.

    Args:
        cfg: AppConfig.
        config_path: YAML config path used for embedding service auto-start.
        mode: Optional indexing mode. If set, overrides legacy flags:
            - ``full``: index all docs; purge existing chunks per doc_id first.
            - ``incremental``: index only changed docs (by doc_hash + mtime); purge per doc_id first.
            - ``purge-missing``: delete doc_ids that no longer exist in vault; no indexing.
        limit: Optional number of markdown docs to process.
        rebuild: If true, drop and recreate collection before indexing.
        rebuild_hard: If true, delete all files under chroma_db_dir before rebuilding.
        purge_doc: If true, delete existing chunks for same doc_id before upsert.
        enable_fts: If true, also index chunks into a local SQLite FTS5 database.
        fts_db_path: Optional path to the FTS5 sqlite database.

    Returns:
        IndexStats.
    """
    md_files = _iter_markdowns(cfg.paths.markdown_vault_dir)
    if limit is not None:
        md_files = md_files[: int(limit)]

    total_docs = len(md_files)
    if total_docs == 0:
        logger.info("No markdown files found in: %s", cfg.paths.markdown_vault_dir)
        return IndexStats(total_docs=0, total_chunks=0, succeeded_docs=0, failed_docs=0, skipped_docs=0)

    state_path = _state_path(cfg)
    if rebuild_hard:
        chroma_dir = cfg.paths.chroma_db_dir
        if chroma_dir.exists():
            for p in sorted(chroma_dir.iterdir(), reverse=True):
                try:
                    if p.is_dir():
                        for sub in p.rglob("*"):
                            if sub.is_file() or sub.is_symlink():
                                sub.unlink(missing_ok=True)  # type: ignore[arg-type]
                        # remove empty dirs bottom-up
                        for subdir in sorted([d for d in p.rglob("*") if d.is_dir()], reverse=True):
                            subdir.rmdir()
                        p.rmdir()
                    else:
                        p.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception as e:
                    logger.warning("Failed to delete chroma path: %s | %s", p, e)

    store = ChromaStore(cfg=cfg)
    fts: Optional[FtsStore] = None
    if enable_fts:
        fts = FtsStore(db_path=Path(fts_db_path) if fts_db_path is not None else default_fts_db_path(cfg))
    # Document-level metadata (documents.document_profile) also lives in the same FTS sqlite.
    doc_meta = FtsStore(db_path=Path(fts_db_path) if fts_db_path is not None else default_fts_db_path(cfg))
    if rebuild or rebuild_hard:
        logger.info("Rebuild enabled. Resetting collection: %s", store.collection_name)
        store.reset_collection()

    mode_s = (mode or "").strip().lower() if mode is not None else ""
    if mode_s == "purge-missing":
        vault_doc_ids = set()
        for md_path in md_files:
            try:
                doc = load_markdown_doc(md_path)
                vault_doc_ids.add(str(doc.doc_id))
            except Exception as e:
                logger.warning("Failed to read markdown for purge-missing: %s | %s", md_path, e)
        chroma_doc_ids = store.list_unique_doc_ids()
        missing = sorted(d for d in chroma_doc_ids if d and d not in vault_doc_ids)
        deleted_total = 0
        for did in missing:
            try:
                deleted_total += int(store.delete_by_doc_id(did))
            except Exception as e:
                logger.warning("Failed to purge missing doc_id=%s | %s", did, e)
        logger.info("purge-missing done. missing_doc_ids=%s deleted_chunks=%s", len(missing), deleted_total)
        return IndexStats(
            total_docs=total_docs,
            total_chunks=0,
            succeeded_docs=0,
            failed_docs=0,
            skipped_docs=total_docs,
        )

    if mode_s in {"full", "incremental"}:
        purge_doc = True

    try:
        embedder = get_embedder(config_path=Path(config_path))
    except Exception as e:
        logger.error("%s", e)
        return IndexStats(total_docs=total_docs, total_chunks=0, succeeded_docs=0, failed_docs=total_docs, skipped_docs=0)

    total_chunks = 0
    succeeded_docs = 0
    failed_docs = 0
    skipped_docs = 0

    # Use ASCII progress bar for better Windows console compatibility.
    for md_path in tqdm(md_files, desc="index_markdown", ascii=True):
        try:
            doc = load_markdown_doc(md_path)
            doc_hash = hashlib.sha1((doc.body or "").encode("utf-8", errors="ignore")).hexdigest()
            try:
                source_md_mtime_ns = int(doc.source_md.resolve().stat().st_mtime_ns)
            except Exception:
                source_md_mtime_ns = None

            if mode_s == "incremental":
                marker = store.get_doc_marker(str(doc.doc_id))
                if marker:
                    old_hash = str(marker.get("doc_hash") or "")
                    old_mtime = marker.get("source_md_mtime_ns")
                    try:
                        old_mtime_i = int(old_mtime) if old_mtime is not None else None
                    except Exception:
                        old_mtime_i = None
                    if old_hash and old_hash == doc_hash and old_mtime_i is not None and source_md_mtime_ns == old_mtime_i:
                        skipped_docs += 1
                        _append_state(
                            state_path,
                            {
                                "status": "skipped",
                                "doc_id": str(doc.doc_id),
                                "md_path": str(md_path.resolve()),
                                "reason": "unchanged_doc_hash_and_mtime",
                            },
                        )
                        continue

            # Best-effort: compute and persist document_profile into FTS SQLite documents table.
            try:
                raw_md = Path(doc.source_md).read_text(encoding="utf-8", errors="ignore")
                prof = build_document_profile(project_root=cfg.project_root, md_text=raw_md, timeout_s=120.0)
                if prof:
                    doc_meta.upsert_document_profile(
                        doc_id=str(doc.doc_id),
                        document_profile=prof,
                        updated_at=now_iso8601(),
                    )
            except Exception as e:
                logger.warning("document_profile build failed doc_id=%s md=%s | %s", doc.doc_id, md_path, e)

            chunks = chunk_markdown(doc, cfg.chunking)
            if not chunks:
                skipped_docs += 1
                _append_state(
                    state_path,
                    {
                        "status": "skipped",
                        "doc_id": str(doc.doc_id),
                        "md_path": str(md_path.resolve()),
                        "reason": "no_chunks",
                    },
                )
                continue

            if purge_doc:
                deleted = store.delete_by_doc_id(doc.doc_id)
                if deleted:
                    logger.info("Purged %s existing chunks for doc_id=%s", deleted, doc.doc_id)
                if fts is not None:
                    try:
                        fts.upsert_chunks(rows=[], purge_doc_ids=[str(doc.doc_id)])
                    except Exception as e:
                        logger.warning("FTS purge failed doc_id=%s | %s", doc.doc_id, e)

            ids: List[str] = []
            docs: List[str] = []
            metas: List[dict] = []

            created_at = now_iso8601()
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
            if fts is not None:
                try:
                    fts_rows = [
                        (c.chunk_id, c.doc_id, doc.short_name, c.section, c.text)
                        for c in chunks
                        if c.chunk_id and c.text
                    ]
                    fts_metas = [
                        (c.chunk_id, str(c.source_md.resolve()), int(c.chunk_index), created_at) for c in chunks
                    ]
                    fts.upsert_chunks(rows=fts_rows, metas=fts_metas)
                except Exception as e:
                    logger.warning("FTS upsert failed doc_id=%s | %s", doc.doc_id, e)

            total_chunks += len(chunks)
            succeeded_docs += 1
            _append_state(
                state_path,
                {
                    "status": "ok",
                    "doc_id": str(doc.doc_id),
                    "short_name": str(doc.short_name),
                    "md_path": str(md_path.resolve()),
                    "chunks": int(len(chunks)),
                },
            )
        except Exception as e:
            failed_docs += 1
            logger.error("Index failed: %s | %s", md_path, e)
            _append_state(
                state_path,
                {
                    "status": "failed",
                    "md_path": str(md_path.resolve()),
                    "error": str(e),
                },
            )
            continue

    return IndexStats(
        total_docs=total_docs,
        total_chunks=total_chunks,
        succeeded_docs=succeeded_docs,
        failed_docs=failed_docs,
        skipped_docs=skipped_docs,
    )

