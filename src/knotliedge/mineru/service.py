from __future__ import annotations

import atexit
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from knotliedge.logging_utils.setup import setup_logging
from knotliedge.mineru.http_client import health

logger = setup_logging()


@dataclass
class MinerUServiceState:
    api_url: str
    pid: int
    started_at: float
    enable_vlm_preload: bool
    host: str
    port: int


_lock = Lock()
_proc: Optional[subprocess.Popen[str]] = None
_state: Optional[MinerUServiceState] = None
_log_fh: Optional[object] = None


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _mineru_api_exe() -> str:
    # Prefer PATH; fallback to current python env Scripts.
    exe = "mineru-api"
    try:
        import shutil
        import sys

        hit = shutil.which(exe)
        if hit:
            return hit
        py = Path(sys.executable)
        candidate = py.parent / "Scripts" / "mineru-api.exe"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return exe


def start_service(
    *,
    enable_vlm_preload: bool = True,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    startup_timeout_s: int = 600,
    fallback_disable_preload: bool = True,
    env_overrides: Optional[Dict[str, str]] = None,
    log_dir: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> MinerUServiceState:
    """Start MinerU FastAPI service if not running.

    ``mineru-api`` writes parse caches under the process working directory; setting
    ``work_dir`` keeps those artifacts out of the repository root.

    Args:
        enable_vlm_preload: Whether to enable VLM preload in mineru-api.
        host: Bind host.
        port: Bind port; if omitted, a free localhost port is chosen.
        startup_timeout_s: Seconds to wait for ``/health`` before failing.
        fallback_disable_preload: If true, retry once with preload disabled on early exit.
        env_overrides: Extra environment variables for the child process.
        log_dir: Directory for mineru-api log files. Defaults to ``Path.cwd() / ".knotliedge"``.
        work_dir: Process working directory for mineru-api. Defaults to
            ``Path.cwd() / ".knotliedge" / "mineru_api_workdir"``.

    Returns:
        ``MinerUServiceState`` describing the running service.
    """
    global _proc, _state
    with _lock:
        if _proc is not None and _proc.poll() is None and _state is not None:
            return _state

        p = int(port) if port is not None else _pick_free_port()
        api_url = f"http://{host}:{p}"

        cmd = [
            _mineru_api_exe(),
            "--host",
            host,
            "--port",
            str(p),
            "--enable-vlm-preload",
            "true" if enable_vlm_preload else "false",
        ]
        logger.info("Starting mineru-api: %s", " ".join(cmd))

        effective_env = os.environ.copy()
        if env_overrides:
            effective_env.update({str(k): str(v) for k, v in env_overrides.items() if v is not None})

        use_log_dir = (log_dir or (Path.cwd() / ".knotliedge")).resolve()
        use_log_dir.mkdir(parents=True, exist_ok=True)
        use_work_dir = (
            work_dir.resolve()
            if work_dir is not None
            else (Path.cwd() / ".knotliedge" / "mineru_api_workdir").resolve()
        )
        use_work_dir.mkdir(parents=True, exist_ok=True)
        log_path = use_log_dir / f"mineru-api-{p}.log"
        global _log_fh
        if _log_fh is not None:
            try:
                _log_fh.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            _log_fh = None
        _log_fh = open(log_path, "a", encoding="utf-8", errors="replace")
        logger.info("mineru-api logs: %s", log_path)
        logger.info("mineru-api cwd: %s", use_work_dir)

        _proc = subprocess.Popen(
            cmd,
            cwd=str(use_work_dir),
            stdout=_log_fh,  # type: ignore[arg-type]
            stderr=_log_fh,  # type: ignore[arg-type]
            text=True,
            encoding="utf-8",
            errors="replace",
            env=effective_env,
        )
        _state = MinerUServiceState(
            api_url=api_url,
            pid=int(_proc.pid or -1),
            started_at=time.time(),
            enable_vlm_preload=bool(enable_vlm_preload),
            host=host,
            port=p,
        )

    # Wait until healthy (outside lock)
    deadline = time.time() + float(startup_timeout_s)
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        if _proc is not None and _proc.poll() is not None:
            msg = f"mineru-api exited early: rc={_proc.returncode} (see log file in .knotliedge/)"
            if enable_vlm_preload and fallback_disable_preload:
                logger.warning("%s. Falling back to disable preload.", msg)
                # reset and retry once without preload
                with _lock:
                    _proc = None
                    _state = None
                return start_service(
                    enable_vlm_preload=False,
                    host=host,
                    port=p,
                    startup_timeout_s=startup_timeout_s,
                    fallback_disable_preload=False,
                    env_overrides=env_overrides,
                    log_dir=log_dir,
                    work_dir=work_dir,
                )
            raise RuntimeError(msg)
        try:
            _ = health(api_url=api_url, timeout_s=2)
            return _state  # type: ignore[return-value]
        except Exception as e:
            last_err = e
            time.sleep(0.5)

    raise RuntimeError(f"mineru-api startup timeout: {api_url} | last_err={last_err}")


def status() -> Dict[str, Any]:
    global _proc, _state
    with _lock:
        if _proc is None or _state is None:
            return {"running": False}
        running = _proc.poll() is None
        return {
            "running": bool(running),
            "pid": int(_state.pid),
            "api_url": _state.api_url,
            "enable_vlm_preload": bool(_state.enable_vlm_preload),
            "host": _state.host,
            "port": int(_state.port),
            "started_at": float(_state.started_at),
        }


def stop_service() -> Dict[str, Any]:
    global _proc, _state, _log_fh
    with _lock:
        if _proc is None:
            _state = None
            return {"stopped": False, "reason": "not_running"}
        proc = _proc
        st = _state
        _proc = None
        _state = None
        fh = _log_fh
        _log_fh = None

    try:
        proc.terminate()
        proc.wait(timeout=10)
        try:
            if fh is not None:
                fh.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        return {"stopped": True, "pid": int(st.pid) if st else None}
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            if fh is not None:
                fh.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        return {"stopped": True, "pid": int(st.pid) if st else None, "forced": True}


atexit.register(stop_service)

