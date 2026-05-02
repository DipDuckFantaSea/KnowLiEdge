from __future__ import annotations

import argparse
from pathlib import Path

from knotliedge.citation_graph.expand import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_PER_NODE_CITES,
    DEFAULT_SLEEP_S,
    run_expand,
)
from knotliedge.logging_utils.setup import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Expand OpenAlex citation subgraph into citations.sqlite3 (depth-limited BFS)."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--db", type=str, default=None, help="Override citations.sqlite3 path.")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="BFS max depth (default 2).")
    parser.add_argument(
        "--per-node-cites",
        type=int,
        default=DEFAULT_PER_NODE_CITES,
        help="Max citing works fetched per node.",
    )
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_S, help="Sleep between HTTP calls (seconds).")
    parser.add_argument("--seed-limit", type=int, default=2000, help="Max DOI seeds from edges + FTS ids.")
    parser.add_argument(
        "--seed-work-id",
        action="append",
        default=[],
        help="Extra OpenAlex work id/url seeds (can be passed multiple times).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from SQLite checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Only list seeds, no writes.")
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Stage candidates into HITL tables only (no writes to openalex_works/openalex_cite_edges).",
    )
    parser.add_argument(
        "--stage-run-id",
        type=str,
        default=None,
        help="Optional staging run id (recommended when using --stage-only).",
    )
    args = parser.parse_args()

    run_expand(
        config_path=Path(args.config),
        citation_db=Path(args.db) if args.db else None,
        max_depth=int(args.max_depth),
        per_node_cites=int(args.per_node_cites),
        sleep_s=float(args.sleep),
        seed_limit=int(args.seed_limit),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        stage_only=bool(args.stage_only),
        stage_run_id=str(args.stage_run_id) if args.stage_run_id else None,
        seed_work_ids=list(args.seed_work_id or []),
    )


if __name__ == "__main__":
    main()
    raise SystemExit(0)

import argparse
from pathlib import Path

from knotliedge.citation_graph.expand import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_PER_NODE_CITES,
    DEFAULT_SLEEP_S,
    run_expand,
)
from knotliedge.logging_utils.setup import setup_logging


def _noop_deleted_legacy() -> None:
    return None


def _work_to_record(
    work: Dict[str, Any],
    *,
    seed_kind: Optional[str],
    depth_seen: int,
    updated_at: str,
) -> Optional[OpenAlexWorkRecord]:
    wid = normalize_openalex_work_id(str(work.get("id") or ""))
    if not wid:
        return None
    inv = work.get("abstract_inverted_index")
    abstract = restore_abstract_from_inverted_index(inv) or None
    py = work.get("publication_year")
    yi: Optional[int] = int(py) if isinstance(py, int) else None
    cbc = work.get("cited_by_count")
    ci: Optional[int] = int(cbc) if isinstance(cbc, int) else None
    title = work.get("title")
    ts = str(title).strip() if isinstance(title, str) else None
    return OpenAlexWorkRecord(
        work_id=wid,
        doi=_doi_from_work(work),
        title=ts,
        publication_year=yi,
        host_venue_display_name=_venue_name(work),
        authors_json=_authors_json(work),
        abstract=abstract,
        cited_by_count=ci,
        seed_kind=seed_kind,
        depth_seen=int(depth_seen),
        updated_at=updated_at,
    )


def _collect_seeds_from_fts(fts_path: Path, *, limit: int) -> List[str]:
    if not fts_path.is_file():
        return []
    out: List[str] = []
    seen: Set[str] = set()
    try:
        con = sqlite3.connect(str(fts_path))
        con.row_factory = sqlite3.Row
        for row in con.execute(
            """
            SELECT DISTINCT openalex_id FROM documents
            WHERE openalex_id IS NOT NULL AND trim(openalex_id) != ''
            LIMIT ?;
            """,
            (int(limit),),
        ):
            raw = str(row["openalex_id"] or "").strip()
            n = normalize_openalex_work_id(raw)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        con.close()
    except Exception as e:
        logger.warning("FTS seed collection failed | path=%s | %s", fts_path, e)
    return out


def run_expand(
    *,
    config_path: Path,
    citation_db: Optional[Path],
    max_depth: int,
    per_node_cites: int,
    sleep_s: float,
    seed_limit: int,
    resume: bool,
    dry_run: bool,
    seed_work_ids: Optional[List[str]] = None,
) -> None:
    cfg = load_app_config(Path(config_path))
    mailto = (cfg.openalex.mailto or "").strip()
    if not mailto:
        raise ValueError("config openalex.mailto is required (OpenAlex Polite Pool).")

    cite_db = Path(citation_db) if citation_db is not None else default_citation_db_path(cfg)
    cg = CitationGraphStore(db_path=cite_db)
    oa = OpenAlexCitationStore(db_path=cite_db)

    dois_from_edges = cg.list_doi_reference_seeds(limit=seed_limit)
    fts_path = default_fts_db_path(cfg)
    fts_seeds = _collect_seeds_from_fts(fts_path, limit=seed_limit)
    extra_work_seeds: List[str] = []
    for x in seed_work_ids or []:
        n = normalize_openalex_work_id(str(x).strip())
        if n:
            extra_work_seeds.append(n)

    if dry_run:
        logger.info(
            "dry-run | doi_seeds=%s fts_openalex_seeds=%s extra_work_seeds=%s",
            len(dois_from_edges),
            len(fts_seeds),
            len(extra_work_seeds),
        )
        for d in dois_from_edges[:20]:
            logger.info("doi_seed | %s", d)
        for w in fts_seeds[:20]:
            logger.info("fts_seed | %s", w)
        for w in extra_work_seeds[:20]:
            logger.info("extra_work_seed | %s", w)
        return

    session = requests.Session()
    session.headers.update({"User-Agent": f"knotliedge-expand-citation/0.1 mailto:{mailto}"})
    api_key = (cfg.openalex.api_key or "").strip() or None
    base_url = "https://api.openalex.org"
    verify_ssl = bool(DEFAULT_VERIFY_SSL)

    seeds: List[str] = []
    for doi in dois_from_edges:
        wid = _resolve_work_id_from_doi(
            session=session,
            doi=doi,
            mailto=mailto,
            api_key=api_key,
            base_url=base_url,
            timeout_s=20,
            verify_ssl=verify_ssl,
        )
        if wid:
            seeds.append(wid)
        time.sleep(float(sleep_s))

    for wid in fts_seeds:
        if wid not in seeds:
            seeds.append(wid)
    for wid in extra_work_seeds:
        if wid not in seeds:
            seeds.append(wid)

    seeds = list(dict.fromkeys(seeds))
    logger.info("Collected %s seed work ids", len(seeds))

    done: Set[str] = set()
    q: Deque[Tuple[str, int, str]] = deque()
    if resume:
        blob = oa.get_checkpoint(CHECKPOINT_KEY) or {}
        raw_p = blob.get("processed")
        if isinstance(raw_p, list):
            for x in raw_p:
                if isinstance(x, str) and x.strip():
                    nw = normalize_openalex_work_id(x.strip())
                    if nw:
                        done.add(nw)
        raw_q = blob.get("queue")
        if isinstance(raw_q, list):
            for item in raw_q:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    w, d, k = item[0], item[1], item[2]
                    if isinstance(w, str) and isinstance(d, int) and isinstance(k, str):
                        nw = normalize_openalex_work_id(w)
                        if nw:
                            q.append((nw, d, k))
    for s in seeds:
        nw = normalize_openalex_work_id(s)
        if nw and nw not in done:
            q.append((nw, 0, "seed"))

    while q:
        work_id, depth, seed_kind = q.popleft()
        if work_id in done:
            continue
        if depth > int(max_depth):
            continue
        ts = now_iso8601()
        work = _fetch_work(
            session=session,
            work_id=work_id,
            mailto=mailto,
            api_key=api_key,
            base_url=base_url,
            timeout_s=DEFAULT_TIMEOUT_S,
            verify_ssl=verify_ssl,
        )
        time.sleep(float(sleep_s))
        if not work:
            logger.warning("OpenAlex fetch returned empty; marking work done to avoid infinite re-queue | work_id=%s", work_id)
            done.add(work_id)
            continue
        rec = _work_to_record(work, seed_kind=seed_kind, depth_seen=depth, updated_at=ts)
        if rec:
            oa.upsert_work(rec)
        done.add(work_id)

        refs = work.get("referenced_works") or []
        edges: List[OpenAlexCiteEdge] = []
        if isinstance(refs, list):
            for r in refs:
                if not isinstance(r, str):
                    continue
                child = normalize_openalex_work_id(r.strip())
                if not child:
                    continue
                edges.append(OpenAlexCiteEdge(src_work_id=work_id, dst_work_id=child, source="referenced_works", created_at=ts))
                if depth < int(max_depth):
                    q.append((child, depth + 1, "referenced"))

        oa.upsert_edges(edges)

        if depth < int(max_depth):
            tail = _tail_from_work_url(work_id)
            if tail:
                citing = _fetch_citing_works(
                    session=session,
                    cited_tail=tail,
                    mailto=mailto,
                    api_key=api_key,
                    base_url=base_url,
                    per_page=int(per_node_cites),
                    timeout_s=DEFAULT_TIMEOUT_S,
                    verify_ssl=verify_ssl,
                )
                time.sleep(float(sleep_s))
                cite_edges: List[OpenAlexCiteEdge] = []
                for cw in citing:
                    cid = normalize_openalex_work_id(str(cw.get("id") or ""))
                    if not cid:
                        continue
                    cite_edges.append(
                        OpenAlexCiteEdge(src_work_id=cid, dst_work_id=work_id, source="cites_filter", created_at=ts)
                    )
                    crec = _work_to_record(cw, seed_kind="citing", depth_seen=depth + 1, updated_at=ts)
                    if crec:
                        oa.upsert_work(crec)
                    if depth + 1 <= int(max_depth):
                        q.append((cid, depth + 1, "citing"))
                oa.upsert_edges(cite_edges)

        oa.set_checkpoint(
            CHECKPOINT_KEY,
            {
                "version": 1,
                "processed": sorted(done),
                "queue": [[w, d, k] for w, d, k in q],
            },
        )

    logger.info("Expansion finished | done_works=%s | db=%s", len(done), cite_db)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Expand OpenAlex citation subgraph into citations.sqlite3 (depth-limited BFS).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--db", type=str, default=None, help="Override citations.sqlite3 path.")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="BFS max depth (default 2).")
    parser.add_argument("--per-node-cites", type=int, default=DEFAULT_PER_NODE_CITES, help="Max citing works fetched per node.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_S, help="Sleep between HTTP calls (seconds).")
    parser.add_argument("--seed-limit", type=int, default=2000, help="Max DOI seeds from edges + FTS ids.")
    parser.add_argument(
        "--seed-work-id",
        action="append",
        default=[],
        help="Extra OpenAlex work id/url seeds (can be passed multiple times).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from SQLite checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Only list seeds, no writes.")
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Stage candidates into HITL tables only (no writes to openalex_works/openalex_cite_edges).",
    )
    parser.add_argument(
        "--stage-run-id",
        type=str,
        default=None,
        help="Optional staging run id (recommended when using --stage-only).",
    )
    args = parser.parse_args()
    run_expand(
        config_path=Path(args.config),
        citation_db=Path(args.db) if args.db else None,
        max_depth=int(args.max_depth),
        per_node_cites=int(args.per_node_cites),
        sleep_s=float(args.sleep),
        seed_limit=int(args.seed_limit),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        stage_only=bool(args.stage_only),
        stage_run_id=str(args.stage_run_id) if args.stage_run_id else None,
        seed_work_ids=list(args.seed_work_id or []),
    )


if __name__ == "__main__":
    main()
