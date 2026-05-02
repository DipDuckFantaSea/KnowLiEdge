from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import yaml

from knotliedge.config.types import AppConfig
from knotliedge.storage.fts_store import default_fts_db_path


@dataclass(frozen=True)
class LocalDocRef:
    doc_id: str
    short_name: str
    md_path: str


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def normalize_doi(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    s = s.replace("DOI:", "").replace("doi:", "").strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE).strip()
    return s.casefold()


def normalize_openalex_id(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.startswith("https://openalex.org/"):
        tail = s[len("https://openalex.org/") :].strip()
        if tail:
            return f"https://openalex.org/{tail}"
        return ""
    if s.startswith("W") and len(s) > 1:
        return f"https://openalex.org/{s}"
    return s


def normalize_title_key(raw: str) -> str:
    s = str(raw or "").strip().casefold()
    if not s:
        return ""
    s = _SPACE_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def iter_vault_markdown_paths(vault_dir: Path) -> Iterator[Path]:
    root = Path(vault_dir).resolve()
    if not root.exists() or not root.is_dir():
        return
    for p in root.rglob("*.md"):
        if p.is_file():
            yield p


def read_frontmatter(md_path: Path) -> Dict[str, object]:
    p = Path(md_path).resolve()
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    if not raw.startswith("---"):
        return {}
    end = raw.find("\n---", 3)
    if end == -1:
        return {}
    block = raw[3:end]
    try:
        obj = yaml.safe_load(block) or {}
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return obj


def build_local_presence_index(cfg: AppConfig) -> Tuple[Dict[str, LocalDocRef], Dict[str, LocalDocRef], Dict[str, LocalDocRef]]:
    """Build indexes from the local Markdown vault and FTS documents metadata.

    Returns:
        (openalex_id_map, doi_map, title_map)
    """

    openalex_id_map: Dict[str, LocalDocRef] = {}
    doi_map: Dict[str, LocalDocRef] = {}
    title_map: Dict[str, LocalDocRef] = {}

    vault = Path(cfg.paths.markdown_vault_dir).resolve()
    for md in iter_vault_markdown_paths(vault):
        fm = read_frontmatter(md)
        doc_id = str(fm.get("doc_id") or "").strip()
        short_name = str(fm.get("short_name") or "").strip()
        if not doc_id:
            continue
        ref = LocalDocRef(
            doc_id=doc_id,
            short_name=short_name,
            md_path=str(md),
        )

        oa = normalize_openalex_id(str(fm.get("openalex_id") or ""))
        if oa and oa not in openalex_id_map:
            openalex_id_map[oa] = ref

        doi = normalize_doi(str(fm.get("doi") or ""))
        if doi and doi not in doi_map:
            doi_map[doi] = ref

        title = normalize_title_key(str(fm.get("title") or ""))
        if title and title not in title_map:
            title_map[title] = ref

    # Enrichment writes doc-level meta to FTS sqlite `documents`, which may have DOI/OpenAlex even if frontmatter doesn't.
    db_path = default_fts_db_path(cfg)
    if db_path.exists():
        try:
            con = sqlite3.connect(str(db_path))
            con.row_factory = sqlite3.Row
        except Exception:
            con = None
        if con is not None:
            try:
                for row in con.execute("SELECT doc_id, doi, openalex_id, openalex_title FROM documents;"):
                    doc_id = str(row["doc_id"] or "").strip()
                    if not doc_id:
                        continue
                    # Best-effort: map doc_id -> a vault md path if it exists.
                    md_guess = str((vault / f"{doc_id}.md").resolve())
                    ref = LocalDocRef(doc_id=doc_id, short_name="", md_path=md_guess)

                    oa = normalize_openalex_id(str(row["openalex_id"] or ""))
                    if oa and oa not in openalex_id_map:
                        openalex_id_map[oa] = ref

                    doi = normalize_doi(str(row["doi"] or ""))
                    if doi and doi not in doi_map:
                        doi_map[doi] = ref

                    title = normalize_title_key(str(row["openalex_title"] or ""))
                    if title and title not in title_map:
                        title_map[title] = ref
            finally:
                try:
                    con.close()
                except Exception:
                    pass

    return openalex_id_map, doi_map, title_map


def build_radar_local_match_map(
    *,
    cfg: AppConfig,
    radar_ids: Iterable[str],
    radar_openalex_ids: Dict[str, str],
    radar_dois: Dict[str, str],
    radar_titles: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """Build mapping radar_id -> local reference info."""

    oa_map, doi_map, title_map = build_local_presence_index(cfg)
    out: Dict[str, Dict[str, str]] = {}

    for rid in radar_ids:
        r = str(rid or "").strip()
        if not r:
            continue

        oa = normalize_openalex_id(str(radar_openalex_ids.get(r) or ""))
        if oa and oa in oa_map:
            ref = oa_map[oa]
            out[r] = {
                "local_doc_id": ref.doc_id,
                "local_short_name": ref.short_name,
                "local_md_path": ref.md_path,
                "local_match": "openalex_id",
            }
            continue

        doi = normalize_doi(str(radar_dois.get(r) or ""))
        if doi and doi in doi_map:
            ref = doi_map[doi]
            out[r] = {
                "local_doc_id": ref.doc_id,
                "local_short_name": ref.short_name,
                "local_md_path": ref.md_path,
                "local_match": "doi",
            }
            continue

        title = normalize_title_key(str(radar_titles.get(r) or ""))
        if title and title in title_map:
            ref = title_map[title]
            out[r] = {
                "local_doc_id": ref.doc_id,
                "local_short_name": ref.short_name,
                "local_md_path": ref.md_path,
                "local_match": "title",
            }
            continue

    return out

