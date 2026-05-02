from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from knotliedge.logging_utils.setup import setup_logging


logger = setup_logging()


@dataclass(frozen=True)
class WatchStateStats:
    records_total: int
    ok_total: int
    failed_total: int
    skipped_total: int
    parse_seconds_mean: Optional[float]
    parse_seconds_p50: Optional[float]
    parse_seconds_p90: Optional[float]
    parse_seconds_p95: Optional[float]
    parse_seconds_max: Optional[float]
    seconds_mean: Optional[float]
    seconds_p50: Optional[float]
    seconds_p90: Optional[float]
    seconds_p95: Optional[float]
    seconds_max: Optional[float]


def _quantile(sorted_vals: List[float], q: float) -> float:
    """Compute quantile with linear interpolation.

    Args:
        sorted_vals: Values sorted ascending.
        q: Quantile in [0, 1].

    Returns:
        Quantile value.
    """
    if not sorted_vals:
        raise ValueError("sorted_vals is empty")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    i = (len(sorted_vals) - 1) * q
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - i) + sorted_vals[hi] * (i - lo)


def compute_stats(path: Path) -> WatchStateStats:
    """Compute summary stats from watch_state.jsonl.

    Args:
        path: Path to jsonl file.

    Returns:
        WatchStateStats.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    parse_seconds: List[float] = []
    seconds: List[float] = []
    total = 0
    ok = 0
    failed = 0
    skipped = 0

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj: Dict[str, Any] = json.loads(line)
        except Exception:
            continue
        total += 1
        st = str(obj.get("status") or "")
        if st == "ok":
            ok += 1
            psec = obj.get("parse_seconds")
            if isinstance(psec, (int, float)):
                parse_seconds.append(float(psec))
            sec = obj.get("seconds")
            if isinstance(sec, (int, float)):
                seconds.append(float(sec))
        elif st == "failed":
            failed += 1
        elif st == "skipped":
            skipped += 1

    if parse_seconds:
        ps = sorted(parse_seconds)
        parse_mean = sum(ps) / len(ps)
        parse_p50 = _quantile(ps, 0.50)
        parse_p90 = _quantile(ps, 0.90)
        parse_p95 = _quantile(ps, 0.95)
        parse_mx = ps[-1]
    else:
        parse_mean = parse_p50 = parse_p90 = parse_p95 = parse_mx = None

    if seconds:
        s = sorted(seconds)
        mean = sum(s) / len(s)
        p50 = _quantile(s, 0.50)
        p90 = _quantile(s, 0.90)
        p95 = _quantile(s, 0.95)
        mx = s[-1]
    else:
        mean = p50 = p90 = p95 = mx = None

    return WatchStateStats(
        records_total=total,
        ok_total=ok,
        failed_total=failed,
        skipped_total=skipped,
        parse_seconds_mean=parse_mean,
        parse_seconds_p50=parse_p50,
        parse_seconds_p90=parse_p90,
        parse_seconds_p95=parse_p95,
        parse_seconds_max=parse_mx,
        seconds_mean=mean,
        seconds_p50=p50,
        seconds_p90=p90,
        seconds_p95=p95,
        seconds_max=mx,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stats from .knotliedge/watch_state.jsonl")
    parser.add_argument(
        "--path",
        type=str,
        default=".knotliedge/watch_state.jsonl",
        help="Path to watch_state.jsonl",
    )
    args = parser.parse_args()
    stats = compute_stats(Path(args.path))
    logger.info(
        "records=%s ok=%s failed=%s skipped=%s",
        stats.records_total,
        stats.ok_total,
        stats.failed_total,
        stats.skipped_total,
    )
    logger.info(
        "parse_seconds_mean=%s p50=%s p90=%s p95=%s max=%s",
        None if stats.parse_seconds_mean is None else round(stats.parse_seconds_mean, 3),
        stats.parse_seconds_p50,
        stats.parse_seconds_p90,
        stats.parse_seconds_p95,
        stats.parse_seconds_max,
    )
    logger.info(
        "seconds_mean=%s p50=%s p90=%s p95=%s max=%s",
        None if stats.seconds_mean is None else round(stats.seconds_mean, 3),
        stats.seconds_p50,
        stats.seconds_p90,
        stats.seconds_p95,
        stats.seconds_max,
    )


if __name__ == "__main__":
    main()

