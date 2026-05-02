from __future__ import annotations

from pathlib import Path

from modelscope import snapshot_download

from knotliedge.logging_utils.setup import setup_logging

logger = setup_logging()


def download_mineru_models() -> Path:
    """Download MinerU core models via ModelScope.

    Returns:
        Absolute path to the downloaded model directory.
    """
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "data" / "00_models" / "pdf_extract_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始从 ModelScope 下载 MinerU 模型（无需 Token）...")
    try:
        downloaded_path = snapshot_download("OpenDataLab/pdf_extract_models", local_dir=str(model_dir))
        logger.info("模型下载成功：%s", downloaded_path)
        return Path(downloaded_path)
    except Exception as e:
        logger.error("模型下载失败：%s", e)
        raise


def main() -> None:
    download_mineru_models()


if __name__ == "__main__":
    main()

