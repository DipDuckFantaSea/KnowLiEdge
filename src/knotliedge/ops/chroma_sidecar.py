from __future__ import annotations

import atexit
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import shutil

_chroma_process: Optional[subprocess.Popen[bytes]] = None


def _is_port_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.2):
            return True
    except OSError:
        return False


def start_chroma_sidecar(*, db_path: str, host: str, port: int, log_path: Optional[str] = None) -> None:
    """Start a local Chroma HTTP daemon as a sidecar (non-blocking).

    If the port is already listening, this is a no-op (assumes an external daemon).
    """

    global _chroma_process
    if _chroma_process and _chroma_process.poll() is None:
        return

    use_host = (host or "").strip() or "localhost"
    use_port = int(port)

    if _is_port_listening(use_host, use_port):
        logging.info("Chroma sidecar not needed (already listening) | host=%s port=%s", use_host, use_port)
        return

    persist = str(Path(db_path).resolve())
    logging.info("正在拉起后台 Chroma 伴生服务 | host=%s port=%s path=%s", use_host, use_port, persist)
    try:
        chroma_exe = shutil.which("chroma")
        if chroma_exe:
            cmd = [chroma_exe, "run", "--path", persist, "--port", str(use_port)]
        else:
            # Fallback: invoke the packaged Rust CLI entry from within Python.
            # chromadb.cli.cli.app() forwards to chromadb_rust_bindings.cli(sys.argv).
            code = (
                "import sys\n"
                "from chromadb.cli.cli import app\n"
                f"sys.argv = ['chroma','run','--path',r'''{persist}''','--port','{use_port}']\n"
                "app()\n"
            )
            cmd = [sys.executable, "-c", code]
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        log_fp = None
        if log_path:
            lp = Path(log_path).resolve()
            lp.parent.mkdir(parents=True, exist_ok=True)
            log_fp = lp.open("ab")
            stdout = log_fp
            stderr = log_fp
        _chroma_process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    except Exception as e:
        logging.error(
            "❌ Chroma 伴生服务启动失败: %s | 请确认已安装 chromadb（含 CLI）或提供 chroma 可执行文件。",
            e,
        )
        raise

    t0 = time.time()
    time.sleep(1.0)
    while time.time() - t0 < 5.0:
        if _is_port_listening(use_host, use_port):
            logging.info("✅ Chroma 伴生服务已就绪 | host=%s port=%s", use_host, use_port)
            return
        if _chroma_process.poll() is not None:
            extra = f" (see log: {log_path})" if log_path else ""
            raise RuntimeError(f"Chroma sidecar exited during startup.{extra}")
        time.sleep(0.25)
    logging.warning("Chroma sidecar may not be ready yet (port not listening) | host=%s port=%s", use_host, use_port)


def stop_chroma_sidecar() -> None:
    """Terminate the sidecar process if we spawned one."""

    global _chroma_process
    if not _chroma_process:
        return
    if _chroma_process.poll() is not None:
        _chroma_process = None
        return
    logging.info("正在清理并关闭后台 Chroma 伴生服务...")
    _chroma_process.terminate()
    try:
        _chroma_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _chroma_process.kill()
    _chroma_process = None
    logging.info("✅ Chroma 伴生服务已安全关闭。")


atexit.register(stop_chroma_sidecar)

