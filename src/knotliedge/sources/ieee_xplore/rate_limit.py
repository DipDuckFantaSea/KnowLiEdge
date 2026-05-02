"""Per-host rate limiting for polite HTTP access."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_ieee_rate_limited_host(hostname: Optional[str]) -> bool:
    """Return True if requests to this host should use IEEE Xplore throttles.

    Args:
        hostname: Parsed hostname (no port), e.g. ``ieeexplore.ieee.org``.

    Returns:
        True when hostname is ``ieee.org`` or a subdomain thereof.
    """
    if not hostname:
        return False
    h = str(hostname).strip().lower().rstrip(".")
    if h == "ieee.org":
        return True
    return h.endswith(".ieee.org")


class HostRateLimiter:
    """Serialize waits between HTTP calls to a specific host class (e.g. IEEE).

    Thread-safe monotonic clock based spacing.
    """

    def __init__(self, *, min_interval_s: float) -> None:
        """Initialize limiter.

        Args:
            min_interval_s: Minimum seconds between consecutive ``wait()`` calls.
        """
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._last: float = 0.0

    @property
    def min_interval_s(self) -> float:
        """Configured minimum spacing in seconds."""
        return self._min_interval_s

    def wait(self) -> None:
        """Block until ``min_interval_s`` has elapsed since the last ``wait()``."""
        if self._min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            need = self._min_interval_s - elapsed
            if need > 0:
                logger.debug("rate_limit sleep %.3fs", need)
                time.sleep(need)
            self._last = time.monotonic()

    def wait_before_url(self, url: str) -> None:
        """Call ``wait()`` only when URL host is IEEE (see ``is_ieee_rate_limited_host``).

        Args:
            url: Full HTTP(S) URL.
        """
        try:
            host = urlparse(url).hostname
        except Exception:
            host = None
        if is_ieee_rate_limited_host(host):
            self.wait()
