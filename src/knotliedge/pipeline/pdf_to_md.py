from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

from knotliedge.config.types import AppConfig
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.parsers.mineru_parser import MinerUNotInstalledError, parse_pdf_with_mineru
from knotliedge.pipeline.doc_id import compute_doc_id
from knotliedge.pipeline.frontmatter import Frontmatter, now_iso8601, write_markdown_with_frontmatter
from knotliedge.pipeline.metadata_extract import extract_paper_metadata
from knotliedge.pipeline.short_name import generate_short_name

logger = setup_logging()


def _state_path(cfg: AppConfig) -> Path:
    return cfg.project_root / ".knotliedge" / "pdf_to_md_state.jsonl"


def _append_state(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class PdfToMdStats:
    """Stats for a PDF->MD batch run."""

    total: int
    succeeded: int
    failed: int
    skipped: int


def _iter_pdfs(raw_pdf_dir: Path) -> Iterable[Path]:
    # Be tolerant on Windows: case-insensitive extensions and nested folders.
    pdfs = list(raw_pdf_dir.rglob("*.pdf")) + list(raw_pdf_dir.rglob("*.PDF"))
    # De-duplicate while keeping determinism
    seen = set()
    out: List[Path] = []
    for p in sorted(pdfs):
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    yield from out


def run_pdf_to_md(cfg: AppConfig, *, limit: Optional[int] = None) -> PdfToMdStats:
    """Run PDF -> Markdown pipeline for all PDFs in raw directory.

    Behavior:
    - Skip if the target markdown file already exists.
    - Any single-file failure is logged and does not stop the batch.

    Args:
        cfg: AppConfig.
        limit: Optional limit of PDFs to process.

    Returns:
        PdfToMdStats.
    """
    pdfs: List[Path] = list(_iter_pdfs(cfg.paths.raw_pdf_dir))
    if limit is not None:
        pdfs = pdfs[: int(limit)]

    total = len(pdfs)
    succeeded = 0
    failed = 0
    skipped = 0

    if total == 0:
        logger.info("No PDFs found in: %s", cfg.paths.raw_pdf_dir)
        return PdfToMdStats(total=0, succeeded=0, failed=0, skipped=0)

    state_path = _state_path(cfg)
    for pdf_path in tqdm(pdfs, desc="pdf_to_md"):
        try:
            doc_id = compute_doc_id(pdf_path)
            md_name = cfg.markdown.vault_filename_pattern.format(doc_id=doc_id)
            md_path = cfg.paths.markdown_vault_dir / md_name

            if md_path.exists():
                skipped += 1
                _append_state(
                    state_path,
                    {
                        "status": "skipped",
                        "doc_id": doc_id,
                        "pdf_path": str(pdf_path.resolve()),
                        "md_path": str(md_path.resolve()),
                        "reason": "md_exists",
                    },
                )
                continue

            # Parse first (MinerU may output assets). We will relocate assets directory by short_name.
            temp_assets_dir = cfg.paths.markdown_assets_dir / doc_id
            result = parse_pdf_with_mineru(pdf_path, assets_dir=temp_assets_dir)

            short_name, _parts = generate_short_name(
                markdown_text=result.markdown_text,
                venue=result.venue,
                pdf_path=pdf_path,
            )
            assets_dir = cfg.paths.markdown_assets_dir / short_name
            if temp_assets_dir.exists() and temp_assets_dir != assets_dir:
                # Best-effort move; if fails, keep temp dir.
                try:
                    assets_dir.parent.mkdir(parents=True, exist_ok=True)
                    temp_assets_dir.rename(assets_dir)
                except Exception:
                    logger.warning("Failed to rename assets dir: %s -> %s", temp_assets_dir, assets_dir)

            fm = Frontmatter(
                doc_id=doc_id,
                short_name=short_name,
                source_pdf=str(pdf_path.resolve()),
                title=result.title or pdf_path.stem,
                authors=extract_paper_metadata(result.markdown_text).authors,
                year=result.year,
                venue=result.venue,
                parsed_at=now_iso8601(),
                parser="mineru",
                version=cfg.markdown.frontmatter_version,
            )
            write_markdown_with_frontmatter(
                output_path=md_path,
                frontmatter=fm,
                markdown_body=result.markdown_text,
            )
            succeeded += 1
            _append_state(
                state_path,
                {
                    "status": "ok",
                    "doc_id": doc_id,
                    "pdf_path": str(pdf_path.resolve()),
                    "md_path": str(md_path.resolve()),
                    "short_name": short_name,
                },
            )
        except MinerUNotInstalledError as e:
            # 明确提示用户手动安装，并停止本批次（因为后续都会失败）
            logger.error(str(e))
            failed = total - succeeded - skipped
            break
        except Exception as e:
            failed += 1
            logger.error("PDF->MD failed: %s | %s", pdf_path, e)
            _append_state(
                state_path,
                {
                    "status": "failed",
                    "pdf_path": str(pdf_path.resolve()),
                    "error": str(e),
                },
            )
            continue

    return PdfToMdStats(total=total, succeeded=succeeded, failed=failed, skipped=skipped)

