import logging
from pathlib import Path
from typing import Optional


def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure root logger with console and optional file handlers.
    """
    logger = logging.getLogger("polymer_pipeline")
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logger configured; log_file=%s", log_file)
    return logger
