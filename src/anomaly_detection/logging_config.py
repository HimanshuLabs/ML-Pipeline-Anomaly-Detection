"""Logging for this pipeline's processes: one format, level from LOG_LEVEL, secrets redacted.

configure_logging() is called once at API startup and at the start of every CLI.
It adds one handler to the root logger, so INFO lines from api.* and
anomaly_detection.* reach the container logs (stderr). Calling it again keeps the
existing handler.

Values of environment variables whose name contains PASSWORD, TOKEN, or SECRET
are replaced by [REDACTED] in every formatted line, tracebacks included.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from typing import IO

LOG_LEVEL_ENV = "LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
REDACTED = "[REDACTED]"

_SECRET_NAME_MARKERS = ("PASSWORD", "TOKEN", "SECRET")
# Shorter values are not redacted: they would also match ordinary words.
_MIN_SECRET_LENGTH = 8

_CONFIGURED_HANDLER: logging.Handler | None = None


def secret_values(environ: Mapping[str, str] | None = None) -> list[str]:
    """Secret values from the environment, longest first."""

    env = os.environ if environ is None else environ
    values = {
        value
        for name, value in env.items()
        if any(marker in name.upper() for marker in _SECRET_NAME_MARKERS)
        and len(value) >= _MIN_SECRET_LENGTH
    }
    return sorted(values, key=len, reverse=True)


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, REDACTED)
    return text


class SecretRedactingFormatter(logging.Formatter):
    """Formats a record, then removes secret values from the whole line."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record), secret_values())


class _StderrHandler(logging.StreamHandler):
    """Writes to the current sys.stderr (which may be replaced after configuration)."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self) -> IO[str]:  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, _value: IO[str]) -> None:
        return None


def _level_from_environment() -> tuple[int, str | None]:
    raw = os.environ.get(LOG_LEVEL_ENV, "").strip()
    if not raw:
        return logging.INFO, None
    level = logging.getLevelNamesMapping().get(raw.upper())
    if level is None:
        return logging.INFO, raw
    return level, None


def configure_logging(*, stream: IO[str] | None = None) -> logging.Handler:
    """Configure the root logger once; returns this pipeline's handler.

    stream: write there instead of stderr (tests).
    """

    global _CONFIGURED_HANDLER

    root = logging.getLogger()
    level, invalid_level = _level_from_environment()

    if _CONFIGURED_HANDLER is not None and _CONFIGURED_HANDLER in root.handlers:
        return _CONFIGURED_HANDLER

    handler: logging.Handler = (
        logging.StreamHandler(stream) if stream is not None else _StderrHandler()
    )
    handler.setFormatter(SecretRedactingFormatter(LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED_HANDLER = handler

    if invalid_level is not None:
        logging.getLogger(__name__).warning(
            "%s=%r is not a logging level; using %s",
            LOG_LEVEL_ENV,
            invalid_level,
            DEFAULT_LOG_LEVEL,
        )
    return handler
