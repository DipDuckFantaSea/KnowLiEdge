"""Rewrite local Markdown image links to match ``paths.markdown_*`` in app config.

Vault layout convention (see ``import_mineru_cli``): images live under
``markdown_assets_dir / {short_name} / images/...`` and ``.md`` files reference them as
``assets/{short_name}/images/...`` (relative to ``markdown_vault_dir``).

This script fixes common drift:

- ``![](images/foo.jpg)`` -> ``![](assets/{short_name}/images/foo.jpg)``
- ``![](assets/wrong_short/...)`` -> ``![](assets/{short_name}/...)``

Optionally copies missing files from ``{vault}/images/`` into the canonical assets tree.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Dict

import yaml

from knotliedge.chunking.md_chunker import split_frontmatter
from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging

logger = logging.getLogger(__name__)

_MD_IMG = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def _normalize_asset_href(inner: str, short_name: str) -> str:
    raw = inner.strip()
    path_only = raw.split('"', 1)[0].strip()
    tail = raw[len(path_only) :]  # optional `` "title" ``
    low = path_only.lower()
    if low.startswith(("http://", "https://", "data:")):
        return inner
    rel = path_only.lstrip("./").replace("\\", "/").lstrip("/")
    if not rel:
        return inner
    sn = short_name.strip().replace("\\", "/")
    expect = f"assets/{sn}/"
    if rel.startswith(expect):
        return inner
    m = re.match(r"^assets/[^/]+/(.+)$", rel)
    if m:
        return f"assets/{sn}/{m.group(1)}" + tail
    if rel.startswith(("images/", "figures/")):
        return f"assets/{sn}/{rel}" + tail
    return inner


def _rewrite_body_image_paths(body: str, short_name: str) -> str:
    def repl(m: re.Match[str]) -> str:
        inner = m.group(2)
        path0 = inner.split('"', 1)[0].strip().split("?", 1)[0]
        suf = Path(path0).suffix.lower()
        if suf not in _IMAGE_EXT:
            return m.group(0)
        new_inner = _normalize_asset_href(inner, short_name)
        return f"{m.group(1)}{new_inner}{m.group(3)}"

    return _MD_IMG.sub(repl, body or "")


def _ensure_asset_on_disk(
    *,
    vault_dir: Path,
    assets_root: Path,
    short_name: str,
    href_inner: str,
) -> bool:
    path_only = href_inner.split('"', 1)[0].strip().split("?", 1)[0].strip()
    rel = path_only.lstrip("./").replace("\\", "/").lstrip("/")
    if not rel.startswith("assets/"):
        return False
    rel_under_assets = rel[len("assets/") :]
    if not rel_under_assets.startswith(f"{short_name}/"):
        return False
    dst = assets_root.joinpath(*rel_under_assets.split("/"))
    if dst.is_file():
        return True
    name = Path(rel_under_assets).name
    for src in (vault_dir / "images" / name, vault_dir / rel):
        try:
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                logger.info("Copied asset | %s -> %s", src, dst)
                return True
        except Exception as e:
            logger.warning("Asset copy attempt failed | src=%s | %s", src, e)
    return False


def _dump_md(fm: Dict[str, object], body: str) -> str:
    fm_yaml = yaml.safe_dump(
        fm,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()
    b = (body or "").lstrip("\n")
    return f"---\n{fm_yaml}\n---\n\n{b}\n"


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Fix vault Markdown image paths per YAML paths.* config.")
    parser.add_argument("--config", type=str, default="sandbox/configs/sandbox.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log changes only; do not write files or copy assets.",
    )
    parser.add_argument(
        "--no-copy-missing",
        action="store_true",
        help="Do not copy missing image files from vault ``images/`` into assets tree.",
    )
    args = parser.parse_args()

    cfg = load_app_config(Path(args.config))
    vault_dir = cfg.paths.markdown_vault_dir.resolve()
    assets_root = cfg.paths.markdown_assets_dir.resolve()

    changed = 0
    scanned = 0
    for md_path in sorted(vault_dir.glob("*.md")):
        scanned += 1
        raw = md_path.read_text(encoding="utf-8", errors="ignore")
        fm, body = split_frontmatter(raw)
        if not isinstance(fm, dict):
            continue
        sn = str(fm.get("short_name") or "").strip()
        if not sn:
            logger.warning("skip (no short_name) | %s", md_path)
            continue
        new_body = _rewrite_body_image_paths(body, sn)
        if new_body == body:
            continue
        if not args.dry_run:
            if not args.no_copy_missing:
                for m in _MD_IMG.finditer(new_body):
                    inner = m.group(2)
                    path0 = inner.split('"', 1)[0].strip().split("?", 1)[0].strip()
                    if Path(path0).suffix.lower() in _IMAGE_EXT:
                        _ensure_asset_on_disk(vault_dir=vault_dir, assets_root=assets_root, short_name=sn, href_inner=inner)
            md_path.write_text(_dump_md(fm, new_body), encoding="utf-8")
        changed += 1
        logger.info("%s | %s", "would update" if args.dry_run else "updated", md_path)

    logger.info("Done. scanned=%s changed=%s dry_run=%s", scanned, changed, bool(args.dry_run))
    print(f"[fix_vault_image_paths] scanned={scanned} changed={changed} dry_run={bool(args.dry_run)}", flush=True)


if __name__ == "__main__":
    main()
