from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from openai import OpenAI

from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.metadata.document_profile import build_profile_draft
from knotliedge.storage.fts_store import FtsStore, default_fts_db_path
from knotliedge.storage.schema import now_iso8601

from knotliedge.llm.dashscope_batch_file import (
    BatchPollConfig,
    create_batch_job,
    download_file_text,
    parse_batch_error_jsonl,
    parse_batch_output_jsonl,
    upload_batch_input_file,
    wait_for_batch,
    write_jsonl,
)

logger = setup_logging(name="knotliedge.batch_document_profile")


DOCUMENT_PROFILE_SYSTEM_PROMPT = (
    "You are a document profiling assistant for academic papers. Your input is a raw profile draft built from a paper's markdown "
    "(TOC + abstract/conclusion excerpts). Your task: compress it into a compact, information-dense Chinese plain text profile while "
    "preserving technical terms, symbols, material names, process nodes, key metrics, and section structure. Do NOT invent facts. "
    "Do NOT add citations. Output MUST be plain text only (no JSON, no markdown fences). Keep it short: target <= 1000 tokens. Prefer the structure:\n"
    "1) 研究问题/动机\n2) 方法/实验设计/建模\n3) 关键结论与指标\n4) 适用边界/局限\n5) 目录骨架(可简化)\n"
    "If information is missing, omit the item."
)


@dataclass(frozen=True)
class RunConfig:
    config_path: Path
    model: str
    enable_thinking: bool
    max_tokens: int
    temperature: float
    batch_size: int
    poll: BatchPollConfig
    output_dir: Path
    force_upsert: bool
    mode: str  # "batch_file" | "batch_chat"
    max_workers: int
    request_timeout_s: float


def _run_id(prefix: str = "batch_document_profile") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"


def _iter_vault_md_files(vault_dir: Path) -> List[Path]:
    md_files = sorted(Path(vault_dir).rglob("*.md"))
    assets_dir = (Path(vault_dir) / "assets").resolve()
    if assets_dir.exists():
        md_files = [p for p in md_files if assets_dir not in p.resolve().parents]
    return md_files


def _read_text(p: Path) -> str:
    return Path(p).read_text(encoding="utf-8", errors="ignore")


def _doc_id_from_path(md_path: Path) -> str:
    # Default vault naming pattern is {doc_id}.md; keep this minimal.
    return md_path.stem.strip()


def _build_request_body(*, model: str, enable_thinking: bool, max_tokens: int, temperature: float, draft: str) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": str(model),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": DOCUMENT_PROFILE_SYSTEM_PROMPT},
            {"role": "user", "content": str(draft)},
        ],
    }
    # DashScope OpenAI-compat non-standard params must be placed into extra_body.
    body["extra_body"] = {"enable_thinking": bool(enable_thinking)}
    return body


def _append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(obj), ensure_ascii=False) + "\n")


def _batched(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    n = max(1, int(batch_size))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _make_client(*, base_url: str) -> OpenAI:
    key = (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY (or OPENAI_API_KEY) in environment/.env")
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("base_url is empty")
    return OpenAI(api_key=key, base_url=base)


def _run_batch_chat_for_group(
    *,
    client: OpenAI,
    run_cfg: RunConfig,
    group: List[Path],
    outputs_dir: Path,
    run_id: str,
    part_idx: int,
    upsert_db: Optional[FtsStore],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Run Batch Chat (long-connection sync) for a small group.

    This is not Batch File; it is the discounted Batch Chat endpoint which keeps the
    connection open until response is ready (up to 3600s).
    """

    from concurrent.futures import ThreadPoolExecutor, as_completed

    cid_to_path: Dict[str, Path] = {}
    requests: Dict[str, Dict[str, Any]] = {}
    for md_path in group:
        doc_id = _doc_id_from_path(md_path)
        cid_to_path[doc_id] = md_path
        raw = _read_text(md_path)
        # Keep payload smaller to reduce connection resets in Batch endpoints.
        draft = build_profile_draft(raw, max_chars_total=6000)
        requests[doc_id] = _build_request_body(
            model=run_cfg.model,
            enable_thinking=run_cfg.enable_thinking,
            max_tokens=run_cfg.max_tokens,
            temperature=run_cfg.temperature,
            draft=draft,
        )

    # Persist inputs for auditing (one JSONL per part).
    input_rows = [{"custom_id": cid, "body": body} for cid, body in requests.items()]
    (outputs_dir.parent / "inputs").mkdir(parents=True, exist_ok=True)
    write_jsonl(path=(outputs_dir.parent / "inputs" / f"{run_id}_part{part_idx:04d}_batch_chat.jsonl"), rows=input_rows)

    ok: Dict[str, str] = {}
    failed: Dict[str, str] = {}

    def _one(cid: str, body: Dict[str, Any]) -> Tuple[str, str]:
        model = body.get("model")
        messages = body.get("messages")
        kwargs = dict(body)
        kwargs.pop("model", None)
        kwargs.pop("messages", None)
        last_err: Optional[BaseException] = None
        for attempt in range(1, 6):
            try:
                resp = client.chat.completions.create(
                    model=str(model),
                    messages=messages,
                    timeout=float(run_cfg.request_timeout_s),
                    **kwargs,
                )
                raw_path = outputs_dir / f"{run_id}_part{part_idx:04d}_{cid}_raw.json"
                try:
                    raw_path.write_text(resp.model_dump_json(), encoding="utf-8")
                except Exception:
                    pass
                content = ""
                try:
                    content = str(resp.choices[0].message.content or "").strip()
                except Exception:
                    content = ""
                return cid, content
            except Exception as e:
                last_err = e
                # Network is flaky in this environment; retry with backoff for connection-ish errors.
                msg = str(e)
                if attempt < 5 and ("Connection error" in msg or "Remote end closed" in msg or "timed out" in msg):
                    time.sleep(min(30.0, 1.6 ** attempt))
                    continue
                err_path = outputs_dir / f"{run_id}_part{part_idx:04d}_{cid}_error.json"
                try:
                    err_path.write_text(str(e), encoding="utf-8")
                except Exception:
                    pass
                raise
        raise RuntimeError(str(last_err or "unknown error"))

    with ThreadPoolExecutor(max_workers=max(1, int(run_cfg.max_workers))) as ex:
        futs = {ex.submit(_one, cid, body): cid for cid, body in requests.items()}
        for fut in as_completed(futs):
            cid = futs[fut]
            try:
                cid2, content = fut.result()
            except Exception as e:
                failed[cid] = str(e)
                continue
            if content:
                ok[cid2] = content
                if upsert_db is not None:
                    upsert_db.upsert_document_profile(
                        doc_id=cid2,
                        document_profile=content,
                        updated_at=now_iso8601(),
                        source=f"batch_chat:{run_cfg.model}:enable_thinking={run_cfg.enable_thinking}",
                    )
            else:
                failed[cid2] = "empty_assistant_content"

    return ok, failed


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _extract_assistant_content(resp_obj: Mapping[str, Any]) -> str:
    choices = resp_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    c0 = choices[0]
    if not isinstance(c0, dict):
        return ""
    msg = c0.get("message")
    if not isinstance(msg, dict):
        return ""
    c = str(msg.get("content") or "").strip()
    if not c:
        return ""
    # Some thinking-enabled models may leak "thinking process" text into content.
    # Keep only the profile section starting at "1) 研究问题/动机" when present.
    anchor = "1) 研究问题/动机"
    idx = c.find(anchor)
    if idx > 0:
        c = c[idx:].lstrip()
    return c


def run_batch_for_docs(
    *,
    run_cfg: RunConfig,
    md_paths: List[Path],
    upsert_db: Optional[FtsStore],
    base_url: str,
) -> Dict[str, Any]:
    """Run Batch File for a list of markdown docs.

    Returns summary dict (counts, batch ids, failures list).
    """

    run_id = _run_id("batch_docprof")
    base_dir = Path(run_cfg.output_dir).resolve()
    inputs_dir = base_dir / "inputs"
    batch_dir = base_dir / "batch"
    poll_dir = base_dir / "poll"
    outputs_dir = base_dir / "outputs"
    summary_dir = base_dir / "summary"

    inputs_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    poll_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    poll_path = poll_dir / f"{run_id}.jsonl"

    client = _make_client(base_url=base_url)

    all_success: Dict[str, str] = {}
    all_failed: Dict[str, str] = {}
    batch_ids: List[str] = []

    total = len(md_paths)
    for bi, group in enumerate(_batched(md_paths, run_cfg.batch_size), start=1):
        if run_cfg.mode == "batch_chat":
            ok, failed = _run_batch_chat_for_group(
                client=client,
                run_cfg=run_cfg,
                group=group,
                outputs_dir=outputs_dir,
                run_id=run_id,
                part_idx=bi,
                upsert_db=upsert_db,
            )
            all_success.update(ok)
            all_failed.update(failed)
            logger.info(
                "Batch Chat part done %s/%s | docs=%s ok=%s failed=%s",
                bi,
                (total + run_cfg.batch_size - 1) // run_cfg.batch_size,
                len(group),
                len(ok),
                len(failed),
            )

            # Retry failed docs (connection errors are common on Batch endpoints).
            retry_rounds = 3
            for r in range(1, retry_rounds + 1):
                if not failed:
                    break
                retry_paths = [Path(run_cfg.config_path).parent.parent]  # dummy to satisfy type checker
                retry_paths = [p for p in group if _doc_id_from_path(p) in failed]
                if not retry_paths:
                    break
                logger.info("Batch Chat retry round %s/%s | docs=%s", r, retry_rounds, len(retry_paths))
                ok2, failed2 = _run_batch_chat_for_group(
                    client=client,
                    run_cfg=run_cfg,
                    group=retry_paths,
                    outputs_dir=outputs_dir,
                    run_id=run_id,
                    part_idx=bi * 100 + r,
                    upsert_db=upsert_db,
                )
                for k, v in ok2.items():
                    all_success[k] = v
                    all_failed.pop(k, None)
                failed = {k: v for k, v in failed2.items()}
                for k, v in failed.items():
                    all_failed[k] = v
            continue

        # Build input JSONL
        custom_rows: List[Dict[str, Any]] = []
        cid_to_doc: Dict[str, str] = {}
        for md_path in group:
            doc_id = _doc_id_from_path(md_path)
            raw = _read_text(md_path)
            draft = build_profile_draft(raw, max_chars_total=6000)
            body = _build_request_body(
                model=run_cfg.model,
                enable_thinking=run_cfg.enable_thinking,
                max_tokens=run_cfg.max_tokens,
                temperature=run_cfg.temperature,
                draft=draft,
            )
            cid = doc_id
            cid_to_doc[cid] = doc_id
            custom_rows.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": body})

        input_jsonl = inputs_dir / f"{run_id}_part{bi:04d}.jsonl"
        write_jsonl(path=input_jsonl, rows=custom_rows)

        # Upload + create batch
        input_file_id = upload_batch_input_file(client=client, jsonl_path=input_jsonl)
        batch = create_batch_job(
            client=client,
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "ds_name": f"document_profile_{run_id}_part{bi:04d}",
                "ds_description": f"doc_profile compress | model={run_cfg.model} enable_thinking={run_cfg.enable_thinking}",
            },
        )
        batch_ids.append(batch.batch_id)
        batch_ids_str = ",".join(batch_ids)
        _append_jsonl(poll_path, {"ts": now_iso8601(), "event": "batch_created", "batch_id": batch.batch_id, "batch_ids": batch_ids_str})
        (batch_dir / f"{run_id}_part{bi:04d}_batch.json").write_text(json.dumps(batch.raw_batch, ensure_ascii=False, indent=2), encoding="utf-8")

        def _on_poll(r, next_sleep_s: float) -> None:
            d = r.raw_batch
            _append_jsonl(
                poll_path,
                {
                    "ts": now_iso8601(),
                    "event": "poll",
                    "batch_id": r.batch_id,
                    "status": r.status,
                    "request_counts": d.get("request_counts"),
                    "next_sleep_s": float(next_sleep_s),
                },
            )

        final = wait_for_batch(client=client, batch_id=batch.batch_id, poll=run_cfg.poll, on_poll=_on_poll)
        _append_jsonl(poll_path, {"ts": now_iso8601(), "event": "batch_terminal", "batch_id": final.batch_id, "status": final.status})

        # Download outputs
        out_text = download_file_text(client=client, file_id=final.output_file_id or "")
        err_text = download_file_text(client=client, file_id=final.error_file_id or "")

        out_path = outputs_dir / f"{run_id}_part{bi:04d}_output.jsonl"
        err_path = outputs_dir / f"{run_id}_part{bi:04d}_error.jsonl"
        _save_text(out_path, out_text)
        _save_text(err_path, err_text)

        out_by_id, _out_rows = parse_batch_output_jsonl(out_text)
        err_by_id, _err_rows = parse_batch_error_jsonl(err_text)

        # Apply results
        for cid, doc_id in cid_to_doc.items():
            if cid in out_by_id:
                resp = out_by_id[cid]
                body = resp.get("body") if isinstance(resp, dict) else None
                prof = _extract_assistant_content(body) if isinstance(body, dict) else ""
                if prof:
                    all_success[doc_id] = prof
                    if upsert_db is not None:
                        upsert_db.upsert_document_profile(
                            doc_id=doc_id,
                            document_profile=prof,
                            updated_at=now_iso8601(),
                            source=f"batch:{run_cfg.model}:enable_thinking={run_cfg.enable_thinking}",
                        )
                    continue
                all_failed[doc_id] = "empty_assistant_content"
                continue
            if cid in err_by_id:
                all_failed[doc_id] = json.dumps(err_by_id[cid], ensure_ascii=False)
            else:
                # Batch could fail or not return this line; mark as missing.
                all_failed[doc_id] = f"missing_result (batch_status={final.status})"

        logger.info(
            "Batch part done %s/%s | docs=%s ok=%s failed=%s",
            bi,
            (total + run_cfg.batch_size - 1) // run_cfg.batch_size,
            len(group),
            len([d for d in group if _doc_id_from_path(d) in all_success]),
            len([d for d in group if _doc_id_from_path(d) in all_failed]),
        )

    summary = {
        "run_id": run_id,
        "model": run_cfg.model,
        "enable_thinking": run_cfg.enable_thinking,
        "total_docs": total,
        "succeeded": len(all_success),
        "failed": len(all_failed),
        "batch_ids": batch_ids,
        "failed_doc_ids": sorted(all_failed.keys()),
    }
    (summary_dir / f"{run_id}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Generate document_profile via DashScope OpenAI-compatible Batch File.")
    p.add_argument("--config", type=str, default=str(Path("sandbox") / "configs" / "sandbox.yaml"), help="YAML config path")
    p.add_argument("--output-dir", type=str, default=str(Path("output") / "batch_document_profile"), help="Where to write records")
    p.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="DashScope compatible-mode base_url")
    p.add_argument("--mode", type=str, default="batch_file", choices=["batch_file", "batch_chat"], help="Execution mode")
    p.add_argument("--model", type=str, default="qwen3.6-plus", help="DashScope model name")
    p.add_argument("--enable-thinking", action="store_true", help="Enable thinking (extra_body.enable_thinking=true)")
    p.add_argument("--disable-thinking", action="store_true", help="Disable thinking (extra_body.enable_thinking=false)")
    p.add_argument("--max-tokens", type=int, default=1800)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=200, help="Docs per batch JSONL file")
    p.add_argument("--poll-initial", type=float, default=10.0)
    p.add_argument("--poll-max", type=float, default=60.0)
    p.add_argument("--poll-backoff", type=float, default=1.7)
    p.add_argument("--limit", type=int, default=0, help="Limit docs (0 = no limit)")
    p.add_argument("--run-10", action="store_true", help="Run on 10 docs (most recently modified)")
    p.add_argument("--dry-run", action="store_true", help="Only build inputs; do not submit batches")
    p.add_argument("--apply-output-jsonl", type=str, default="", help="Apply an existing Batch File output.jsonl to DB (no API call).")
    p.add_argument("--force-upsert", action="store_true", help="Force upsert into FTS documents table")
    p.add_argument("--max-workers", type=int, default=6, help="Batch Chat parallelism (threads)")
    p.add_argument("--request-timeout", type=float, default=3600.0, help="Per-request timeout seconds for Batch Chat")
    args = p.parse_args()

    cfg = load_app_config(Path(args.config))
    vault = cfg.paths.markdown_vault_dir
    md_files = _iter_vault_md_files(vault)
    if args.run_10:
        # Most recently modified 10
        md_files = sorted(md_files, key=lambda x: x.stat().st_mtime_ns if x.exists() else 0, reverse=True)[:10]
    if args.limit and int(args.limit) > 0:
        md_files = md_files[: int(args.limit)]

    if not md_files:
        logger.info("No markdown files found: %s", vault)
        return 0

    enable_thinking = True
    if args.disable_thinking:
        enable_thinking = False
    if args.enable_thinking:
        enable_thinking = True

    run_cfg = RunConfig(
        config_path=Path(args.config),
        model=str(args.model),
        enable_thinking=bool(enable_thinking),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        batch_size=int(args.batch_size),
        poll=BatchPollConfig(
            initial_interval_s=float(args.poll_initial),
            max_interval_s=float(args.poll_max),
            backoff_factor=float(args.poll_backoff),
            max_polls=None,
        ),
        output_dir=Path(args.output_dir),
        force_upsert=bool(args.force_upsert),
        mode=str(args.mode),
        max_workers=int(args.max_workers),
        request_timeout_s=float(args.request_timeout),
    )

    base_dir = Path(run_cfg.output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    base_url = str(args.base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("--base-url is empty")

    # Apply an existing Batch File output JSONL and exit (offline replay).
    apply_out = str(args.apply_output_jsonl or "").strip()
    if apply_out:
        if not args.force_upsert:
            raise ValueError("--apply-output-jsonl requires --force-upsert")
        out_text = Path(apply_out).read_text(encoding="utf-8", errors="ignore")
        out_by_id, _rows = parse_batch_output_jsonl(out_text)
        db = FtsStore(db_path=default_fts_db_path(cfg))
        ok = 0
        for doc_id, resp in out_by_id.items():
            if not isinstance(resp, dict):
                continue
            body = resp.get("body")
            if not isinstance(body, dict):
                continue
            prof = _extract_assistant_content(body)
            if not prof:
                continue
            db.upsert_document_profile(
                doc_id=str(doc_id),
                document_profile=prof,
                updated_at=now_iso8601(),
                source=f"batch_file_replay:{args.model}:enable_thinking={bool(args.enable_thinking and not args.disable_thinking)}",
            )
            ok += 1
        logger.info("applied output jsonl ok=%s path=%s", ok, apply_out)
        return 0

    if args.dry_run:
        run_id = _run_id("dryrun_docprof")
        inputs_dir = base_dir / "inputs"
        rows = []
        for md_path in md_files:
            doc_id = _doc_id_from_path(md_path)
            raw = _read_text(md_path)
            draft = build_profile_draft(raw, max_chars_total=6000)
            body = _build_request_body(
                model=run_cfg.model,
                enable_thinking=run_cfg.enable_thinking,
                max_tokens=run_cfg.max_tokens,
                temperature=run_cfg.temperature,
                draft=draft,
            )
            rows.append({"custom_id": doc_id, "method": "POST", "url": "/v1/chat/completions", "body": body})
        input_jsonl = inputs_dir / f"{run_id}.jsonl"
        write_jsonl(path=input_jsonl, rows=rows)
        logger.info("dry-run wrote input jsonl: %s (rows=%s)", input_jsonl, len(rows))
        return 0

    upsert_db: Optional[FtsStore] = None
    if args.force_upsert:
        upsert_db = FtsStore(db_path=default_fts_db_path(cfg))

    summary = run_batch_for_docs(run_cfg=run_cfg, md_paths=md_files, upsert_db=upsert_db, base_url=base_url)
    logger.info("done: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

