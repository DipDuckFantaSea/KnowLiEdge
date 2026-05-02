from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(level: int = logging.INFO, *, name: str = "knotliedge") -> logging.Logger:
    """Configure and return a project logger.

    Args:
        level: Logging level, e.g. logging.INFO.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    use_rich = os.environ.get("KNOTLIEDGE_RICH_LOG", "1").strip().lower() not in {"0", "false", "no"}

    if use_rich:
        try:
            from rich.console import Console
            from rich.logging import RichHandler

            # Windows note: many invocations run under conda/captured stdout where Unicode box-drawing
            # characters may be mis-encoded. Prefer ASCII-safe rendering for stable visuals.
            console = Console(
                stderr=True,
                force_terminal=True,
                legacy_windows=True,
                emoji=False,
                highlight=False,
                markup=False,
                safe_box=True,
            )
            handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                markup=False,
                rich_tracebacks=True,
            )
            handler.setLevel(level)
            logger.addHandler(handler)
            logger.propagate = False
            return logger
        except Exception:
            # Fall back to plain StreamHandler if rich isn't installed.
            pass

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger

