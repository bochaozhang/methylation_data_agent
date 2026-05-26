"""
Centralized logging configuration for MethyAgent.
"""
import logging
import os
from pathlib import Path

# Default log directory: prefer /workspace (local scratch) over /mnt/results (S3-backed)
_DEFAULT_LOG_FILE = os.environ.get(
    "METHYAGENT_LOG_FILE",
    "/workspace/methyagent_logs/methyagent.log",
)


def get_logger(
    name: str,
    log_file: str = _DEFAULT_LOG_FILE,
    level: str = "INFO",
) -> logging.Logger:
    """
    Get a named logger with both console and file handlers.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_file: Path to the log file. Defaults to /workspace/methyagent_logs/methyagent.log
                  (local scratch, survives the session but not machine restarts).
                  Override via METHYAGENT_LOG_FILE environment variable.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured Logger instance.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional — skip gracefully if path is not writable)
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (PermissionError, OSError):
        # Non-fatal: log to console only if file path is not writable
        pass

    return logger
