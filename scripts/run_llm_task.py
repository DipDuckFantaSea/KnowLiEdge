from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from knotliedge.llm.task_runner import run_task


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an OpenAI-compatible LLM task by task_id.")
    p.add_argument("--config", type=str, default=str(Path("config") / "llm_tasks.json"), help="Path to llm task config JSON.")
    p.add_argument("--task", type=str, required=True, help="Task id in config.tasks")
    p.add_argument("--query", type=str, default="", help="User query injected into the task template")
    p.add_argument("--query-file", type=str, default="", help="Read user query from a UTF-8 text file (preferred for multiline).")
    p.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    p.add_argument(
        "--override-json",
        type=str,
        default="",
        help='Shallow overrides merged into request, e.g. \'{"model":"deepseek-v4-pro","reasoning_effort":"low"}\'',
    )
    p.add_argument("--override-file", type=str, default="", help="Read shallow overrides from a JSON file (preferred on Windows).")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()

    query = str(args.query or "")
    query_file = str(args.query_file or "").strip()
    if query_file:
        query = Path(query_file).read_text(encoding="utf-8", errors="strict")
    if not query.strip():
        raise ValueError("Either --query or --query-file must be provided (non-empty).")

    overrides = None
    override_file = str(args.override_file or "").strip()
    if override_file:
        overrides = json.loads(Path(override_file).read_text(encoding="utf-8-sig", errors="strict"))
        if not isinstance(overrides, dict):
            raise ValueError("--override-file must be a JSON object")
    elif args.override_json:
        overrides = json.loads(args.override_json)
        if not isinstance(overrides, dict):
            raise ValueError("--override-json must be a JSON object")

    out = run_task(
        config_path=Path(args.config),
        task_id=str(args.task),
        user_query=query,
        timeout_s=float(args.timeout),
        overrides=overrides,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

