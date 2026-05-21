"""Structured logging setup."""

import logging
import sys

from common.settings import get_settings


def setup_logging(name: str) -> logging.Logger:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(name)
