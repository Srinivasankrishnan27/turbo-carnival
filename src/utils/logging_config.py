import logging
from typing import Optional

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def init_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """Initialize package-wide logging.

    Call this once from your application entrypoint (tests or CLI) to set
    a default logging configuration used across evaluators and modules.
    """
    fmt = fmt or DEFAULT_FORMAT
    # basicConfig is idempotent for simple setups; users can configure dictConfig
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger().debug("Logging initialized at level %s", logging.getLevelName(level))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger for use in modules.

    Example:
        from src.utils.logging_config import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
