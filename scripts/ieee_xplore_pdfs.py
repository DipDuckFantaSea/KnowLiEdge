"""IEEE Xplore: build DOI manifest (Crossref+OpenAlex) and slow serial PDF download via stamp.jsp."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.sources.ieee_xplore.manifest_builder import build_entries_for_dois
from knotliedge.sources.ieee_xplore.rate_limit import HostRateLimiter
from knotliedge.sources.ieee_xplore.stamp_pdf import download_pdf_via_stamp, merge_mozilla_cookie_file

logger = logging.getLogger(__name__)


def _parse_night_window(spec: str) -> Tuple[int, int]:
    s = spec.strip()
    if "," in s:
        a, b = s.split(",", 1)
    elif "-" in s:
        a, b = s.split("-", 1)
    else:
        raise ValueError("night window must look like 22,7 or 22-7 (start,end local hour)")
    return int(a.strip()), int(b.strip())


def _within_night_window(now: datetime, start_h: int, end_h: int) -> bool:
    h = now.hour
    if start_h == end_h:
        return True
    if start_h < end_h:
        return start_h <= h < end_h
    return h >= start_h or h < end_h


def _read_dois_from_file(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _iter_manifest_lines(path: Path) -> Iterator[dict]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        yield json.loads(s)


def cmd_manifest(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config).resolve()
    cfg = load_app_config(cfg_path)
    dois: List[str] = []
    if args.doi_file:
        dois.extend(_read_dois_from_file(Path(args.doi_file).resolve()))
    for d in args.doi or []:
        dois.append(str(d).strip())
    if not dois:
        logger.error("no DOIs: pass --doi and/or --doi-file")
        return 2
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries = build_entries_for_dois(
        dois,
        mailto=cfg.openalex.mailto,
        openalex_api_key=cfg.openalex.api_key,
        metadata_sleep_s=float(args.metadata_sleep_s),
        between_metadata_sources_sleep_s=float(args.between_metadata_sleep_s),
    )
    if args.dry_run:
        logger.info("dry-run: would write %s rows", len(entries))
        for e in entries[:20]:
            logger.info("%s", e.to_jsonl_line())
        if len(entries) > 20:
            logger.info("... (%s more)", len(entries) - 20)
        return 0
    with out_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.to_jsonl_line() + "\n")
    logger.info("wrote manifest | %s | rows=%s", out_path, len(entries))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config).resolve()
    cfg = load_app_config(cfg_path)
    ix = cfg.ieee_xplore

    if args.night_window:
        sh, eh = _parse_night_window(str(args.night_window))
        if not _within_night_window(datetime.now(), sh, eh):
            logger.error("outside night window %s-%s local time; abort", sh, eh)
            return 3

    try:
        import requests
    except Exception as e:
        logger.error("requests not installed | %s", e)
        return 2

    session = requests.Session()
    session.headers.update({"User-Agent": ix.user_agent})
    session.trust_env = bool(ix.trust_env_for_proxy)
    if args.cookies:
        try:
            merge_mozilla_cookie_file(session, Path(args.cookies).resolve())
        except OSError as e:
            logger.error("cookie file failed | %s", e)
            return 2

    limiter = HostRateLimiter(min_interval_s=ix.min_interval_s)
    raw_dir = cfg.paths.raw_pdf_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, Path]] = []
    if args.arnumber:
        arn = str(args.arnumber).strip()
        out = Path(args.output).resolve() if args.output else (raw_dir / f"ieee_{arn}.pdf")
        tasks.append((arn, out))
    if args.manifest:
        for row in _iter_manifest_lines(Path(args.manifest).resolve()):
            arn = str(row.get("arnumber") or "").strip()
            if not arn.isdigit():
                continue
            name = str(row.get("pdf_filename") or "").strip() or f"ieee_{arn}.pdf"
            out = raw_dir / name
            if str(row.get("status") or "").lower() in {"skip", "done", "ok"}:
                continue
            tasks.append((arn, out))

    if not tasks:
        logger.error("nothing to download (use --arnumber or --manifest with pending rows)")
        return 2

    if args.dry_run:
        for arn, out in tasks:
            logger.info("dry-run: would fetch arnumber=%s -> %s", arn, out)
        return 0

    failed = 0
    for arn, out in tasks:
        try:
            download_pdf_via_stamp(
                session,
                arn,
                out,
                max_retries=ix.max_retries,
                backoff_base_s=ix.backoff_base_s,
                rate_limiter=limiter,
                skip_if_exists=not bool(args.no_skip_existing),
            )
        except Exception as e:
            failed += 1
            logger.error("download failed | arnumber=%s | %s", arn, e)
    return 1 if failed else 0


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="IEEE Xplore manifest + slow PDF fetch (institutional).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_man = sub.add_parser("manifest", help="Resolve DOIs to arnumbers via Crossref+OpenAlex; write JSONL.")
    p_man.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config (paths + openalex + ieee_xplore).")
    p_man.add_argument("--doi", action="append", default=[], help="Repeatable DOI.")
    p_man.add_argument("--doi-file", type=str, default=None, help="Text file: one DOI per line.")
    p_man.add_argument("--out", type=str, required=True, help="Output JSONL path.")
    p_man.add_argument("--metadata-sleep-s", type=float, default=2.0, help="Sleep after each DOI resolution.")
    p_man.add_argument("--between-metadata-sleep-s", type=float, default=1.0, help="Sleep between Crossref and OpenAlex.")
    p_man.add_argument("--dry-run", action="store_true")
    p_man.set_defaults(func=cmd_manifest)

    p_dl = sub.add_parser("download", help="Download PDFs via stamp.jsp (strict IEEE spacing from config).")
    p_dl.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config (paths + openalex + ieee_xplore).")
    p_dl.add_argument("--arnumber", type=str, default=None, help="Single IEEE document id.")
    p_dl.add_argument("--output", type=str, default=None, help="Output path when using single --arnumber.")
    p_dl.add_argument("--manifest", type=str, default=None, help="JSONL from manifest subcommand.")
    p_dl.add_argument("--cookies", type=str, default=None, help="Mozilla-format cookies.txt for session auth.")
    p_dl.add_argument("--night-window", type=str, default=None, help="Only run inside local hours, e.g. 22-7 or 22,7.")
    p_dl.add_argument("--dry-run", action="store_true")
    p_dl.add_argument("--no-skip-existing", action="store_true", help="Re-download even if PDF exists.")
    p_dl.set_defaults(func=cmd_download)

    args = parser.parse_args()
    rc = int(args.func(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
