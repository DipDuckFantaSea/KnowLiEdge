from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from knotliedge.config.types import AppConfig
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.pipeline.doc_id import compute_doc_id
from knotliedge.pipeline.frontmatter import Frontmatter, now_iso8601, write_markdown_with_frontmatter
from knotliedge.pipeline.metadata_extract import extract_paper_metadata
from knotliedge.pipeline.short_name import generate_short_name

logger = setup_logging()


@dataclass(frozen=True)
class ImportStats:
    """Stats for importing MinerU CLI outputs."""

    total_md: int
    succeeded: int
    failed: int
    skipped: int


_MD_LINK_RE = re.compile(r"(?P<prefix>!\[[^\]]*\]\()(?P<path>[^)]+)(?P<suffix>\))")


def _iter_mds(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.rglob("*.md"))


def _find_pdf_for_md(md_path: Path, *, pdf_dir: Optional[Path]) -> Optional[Path]:
    if pdf_dir is None:
        return None
    stem = md_path.stem
    cand1 = pdf_dir / f"{stem}.pdf"
    cand2 = pdf_dir / f"{stem}.PDF"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    # Best-effort: try exact filename match ignoring extension case
    for p in pdf_dir.glob("*.pdf"):
        if p.stem == stem:
            return p
    for p in pdf_dir.glob("*.PDF"):
        if p.stem == stem:
            return p
    return None


def _compute_doc_id_fallback(md_text: str) -> str:
    h = sha1((md_text or "").encode("utf-8", errors="ignore")).hexdigest()
    return h


def _collect_asset_paths(md_text: str) -> List[Path]:
    paths: List[Path] = []
    for m in _MD_LINK_RE.finditer(md_text or ""):
        raw = (m.group("path") or "").strip()
        # strip title part: "path \"title\""
        raw = raw.split('"', 1)[0].strip()
        if not raw:
            continue
        # ignore URLs
        low = raw.lower()
        if low.startswith("http://") or low.startswith("https://") or low.startswith("data:"):
            continue
        # normalize leading ./ and /
        raw = raw.lstrip("./")
        raw = raw.lstrip("/")
        if not raw:
            continue
        p = Path(raw)
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
            paths.append(p)
    # de-dup preserve order
    seen = set()
    out: List[Path] = []
    for p in paths:
        k = str(p).replace("\\", "/").lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _rewrite_asset_links(md_text: str, *, short_name: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        raw = (m.group("path") or "").strip()
        raw_no_title = raw.split('"', 1)[0].strip()
        low = raw_no_title.lower()
        if low.startswith("http://") or low.startswith("https://") or low.startswith("data:") or not raw_no_title:
            return m.group(0)
        rel = raw_no_title.lstrip("./").lstrip("/")
        if not rel:
            return m.group(0)
        new_path = f"assets/{short_name}/{rel}".replace("\\", "/")
        # keep original title part if present
        title_part = ""
        if '"' in raw:
            title_part = raw[raw.find('"') :]
        return f"{m.group('prefix')}{new_path}{title_part}{m.group('suffix')}"

    return _MD_LINK_RE.sub(_repl, md_text or "")


def _copy_assets(
    *,
    md_path: Path,
    md_text: str,
    input_dir: Path,
    assets_root: Path,
    short_name: str,
) -> Tuple[int, int]:
    """Copy referenced assets from input_dir to assets/{short_name}/... and return (found, copied)."""
    rel_paths = _collect_asset_paths(md_text)
    found = 0
    copied = 0
    for rel in rel_paths:
        src = (md_path.parent / rel).resolve()
        if not src.exists():
            src = (input_dir / rel).resolve()
        if not src.exists():
            continue
        found += 1
        dst = assets_root / short_name / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            copied += 1
        except Exception as e:
            logger.warning("Asset copy failed: %s -> %s | %s", src, dst, e)
    return found, copied


def run_import_mineru_cli_outputs(
    cfg: AppConfig,
    *,
    input_dir: Path,
    pdf_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    skip_existing: bool = True,
) -> ImportStats:
    """Import MinerU CLI outputs (.md + assets) into project-standard vault layout.

    Expected CLI outputs (your current case): a directory containing one or more .md files,
    with relative asset references like ``images/<hash>.jpg``.

    Output:
    - markdown_vault_dir/{doc_id}.md (with YAML frontmatter)
    - markdown_assets_dir/{short_name}/<relative_asset_path>
    - rewritten markdown image links: ``images/...`` -> ``assets/{short_name}/images/...``

    Args:
        cfg: AppConfig.
        input_dir: Directory to scan for .md files.
        pdf_dir: Optional directory to match PDFs by md filename stem for stable doc_id.
        limit: Optional limit number of markdown files to import.
        skip_existing: Skip if target {doc_id}.md already exists.

    Returns:
        ImportStats.
    """
    md_files = list(_iter_mds(input_dir))
    if limit is not None:
        md_files = md_files[: int(limit)]

    total = len(md_files)
    ok = 0
    failed = 0
    skipped = 0

    for md_path in md_files:
        try:
            md_text = md_path.read_text(encoding="utf-8", errors="ignore")
            pdf_path = _find_pdf_for_md(md_path, pdf_dir=pdf_dir)
            if pdf_path is not None and pdf_path.exists():
                doc_id = compute_doc_id(pdf_path)
                source_pdf = str(pdf_path.resolve())
                title_fallback = pdf_path.stem
            else:
                doc_id = _compute_doc_id_fallback(md_text)
                source_pdf = ""
                title_fallback = md_path.stem

            out_md_name = cfg.markdown.vault_filename_pattern.format(doc_id=doc_id)
            out_md_path = cfg.paths.markdown_vault_dir / out_md_name
            if skip_existing and out_md_path.exists():
                skipped += 1
                continue

            meta = extract_paper_metadata(md_text)
            short_name, _parts = generate_short_name(markdown_text=md_text, venue=meta.venue, pdf_path=pdf_path or md_path)
            _found, _copied = _copy_assets(
                md_path=md_path,
                md_text=md_text,
                input_dir=input_dir,
                assets_root=cfg.paths.markdown_assets_dir,
                short_name=short_name,
            )
            rewritten = _rewrite_asset_links(md_text, short_name=short_name)

            fm = Frontmatter(
                doc_id=doc_id,
                short_name=short_name,
                source_pdf=source_pdf,
                title=meta.title or title_fallback,
                authors=meta.authors,
                year=meta.year,
                venue=meta.venue,
                parsed_at=now_iso8601(),
                parser="mineru-cli-import",
                version=cfg.markdown.frontmatter_version,
            )
            write_markdown_with_frontmatter(output_path=out_md_path, frontmatter=fm, markdown_body=rewritten)
            ok += 1
        except Exception as e:
            failed += 1
            logger.error("Import failed: %s | %s", md_path, e)
            continue

    return ImportStats(total_md=total, succeeded=ok, failed=failed, skipped=skipped)

