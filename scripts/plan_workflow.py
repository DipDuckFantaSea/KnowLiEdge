from __future__ import annotations

import argparse
from pathlib import Path

from knotliedge.config.load import load_app_config
from knotliedge.logging_utils.setup import setup_logging
from knotliedge.workflow.planning import build_and_write_workflow_plan_markdown

setup_logging()


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a replayable workflow plan (M0 protocol).")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config (default or sandbox).")
    ap.add_argument("--prompt", type=str, required=True, help="User request to plan for.")
    ap.add_argument(
        "--intent",
        type=str,
        default="research",
        help="One of: research|summary|compare|extract_table|translate|other (best-effort).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["fast", "standard"],
        help="Planning mode: fast omits optional graph steps unless hinted; standard includes them.",
    )
    args = ap.parse_args()

    cfg = load_app_config(Path(args.config))
    _plan, plan_md = build_and_write_workflow_plan_markdown(
        cfg=cfg,
        prompt=str(args.prompt),
        intent=str(args.intent),
        mode=str(args.mode),  # type: ignore[arg-type]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

