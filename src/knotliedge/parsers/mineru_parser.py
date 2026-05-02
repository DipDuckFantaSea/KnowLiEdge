from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

from knotliedge.logging_utils.setup import setup_logging
from knotliedge.pipeline.metadata_extract import extract_paper_metadata

logger = setup_logging()


@dataclass(frozen=True)
class MinerUParseResult:
    """Result of parsing a single PDF with MinerU."""

    markdown_text: str
    title: str
    authors_raw: str
    year: Optional[int]
    venue: Optional[str]


class MinerUNotInstalledError(RuntimeError):
    """Raised when MinerU is not available on the system."""


def _require_mineru_cli() -> str:
    exe = shutil.which("mineru")
    if not exe:
        # Fallback: when running with a venv/conda python directly, PATH may not include env Scripts.
        # Try to locate mineru.exe relative to current interpreter.
        py = Path(sys.executable)
        candidate = py.parent / "Scripts" / "mineru.exe"
        if candidate.exists():
            return str(candidate)
        raise MinerUNotInstalledError(
            "未检测到 MinerU CLI（mineru）。需要你手动安装/配置 MinerU，并确保命令 `mineru` 可在终端直接运行。"
        )
    return exe


def parse_pdf_with_mineru(pdf_path: Path, *, assets_dir: Path, timeout_s: int = 600) -> MinerUParseResult:
    """Parse a PDF into Markdown using MinerU CLI.

    注意：本项目阶段一以“可读Markdown优先”。MinerU 的具体参数/输出格式可能因版本不同而变化，
    这里采用“最小可用”的适配：若 MinerU 产出 markdown 文件，则读取其内容；否则抛错。

    Args:
        pdf_path: Input PDF path.
        assets_dir: Directory for assets produced by MinerU.
        timeout_s: Timeout seconds for MinerU CLI execution.

    Returns:
        MinerUParseResult containing markdown and minimal metadata.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_dir = assets_dir / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer invoking MinerU via the *current interpreter* to avoid PATH/user-site pollution
    # (we observed failures loading torch from user-level site-packages on Windows).
    #
    # Equivalent to running `mineru ...`, but ensures it uses the active conda/venv python.
    # The `-s` flag disables user site-packages, preventing accidental import of
    # `C:\\Users\\...\\Python310\\site-packages\\torch` which can trigger WinError 1455.
    try:
        _require_mineru_cli()
    except MinerUNotInstalledError:
        # still raise the same error message
        raise

    # MinerU CLI (current) expects: mineru -p <path> -o <output_dir>
    # Use pipeline backend + txt mode by default to avoid heavy VLM engines and slow OCR.
    cmd = [sys.executable, "-s", "-m", "mineru.cli.client", "-p", str(pdf_path), "-o", str(out_dir), "-b", "pipeline", "-m", "txt"]
    logger.info("MinerU parsing: %s", pdf_path)

    try:
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        # Windows: avoid hard crash when multiple OpenMP runtimes are present.
        # This is a known issue when mixing MKL/Intel OpenMP and other OpenMP-linked libs.
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("OMP_NUM_THREADS", "1")
        # Avoid inheriting a user-level PYTHONPATH that can shadow env packages.
        # Inject project-local dependencies if needed (e.g. pylatexenc) without enabling user site-packages.
        pydeps = (Path(__file__).resolve().parents[3] / ".knotliedge" / "pydeps").resolve()
        if pydeps.exists():
            env["PYTHONPATH"] = str(pydeps)
        else:
            env.pop("PYTHONPATH", None)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        stdout, stderr = proc.communicate(timeout=int(timeout_s))
        if proc.returncode != 0:
            raise RuntimeError(f"MinerU 解析失败: {pdf_path} | {(stderr or '').strip()}")
    except subprocess.TimeoutExpired as e:
        try:
            proc.kill()  # type: ignore[name-defined]
            proc.communicate(timeout=10)  # best-effort drain
        except Exception:
            pass
        raise RuntimeError(f"MinerU 解析超时(>{int(timeout_s)}s): {pdf_path}") from e

    md_candidates = sorted(out_dir.rglob("*.md"))
    if not md_candidates:
        raise RuntimeError(f"MinerU 未产出 Markdown 文件: {pdf_path} | out_dir={out_dir}")

    md_path = md_candidates[0]
    markdown_text = md_path.read_text(encoding="utf-8", errors="ignore")

    meta = extract_paper_metadata(markdown_text)
    title = meta.title or pdf_path.stem
    return MinerUParseResult(
        markdown_text=markdown_text,
        title=title,
        authors_raw=", ".join(meta.authors),
        year=meta.year,
        venue=meta.venue,
    )

