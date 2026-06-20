"""
Structured JSON logging for Voice2Prompt.

Each stage emits per-request JSON log lines with at minimum:
  latency_ms, tokens_in, tokens_out, model_id, stage

Usage:
    from voice2prompt.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("stage_complete", latency_ms=42.3, tokens_out=120)
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, MutableMapping


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        skip = {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
        for key, val in record.__dict__.items():
            if key not in skip:
                payload[key] = val
        return json.dumps(payload)


class _KVAdapter(logging.LoggerAdapter):
    """Allows logger.info("event", key=val, ...) syntax."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        extra = kwargs.pop("extra", {})
        for k in list(kwargs.keys()):
            if k not in ("exc_info", "stack_info", "stacklevel"):
                extra[k] = kwargs.pop(k)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> _KVAdapter:
    base = logging.getLogger(name)
    if not base.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        base.addHandler(handler)
        base.propagate = False
    return _KVAdapter(base, {})


def configure_root(level: str = "INFO", fmt: str = "json") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("voice2prompt")
    root.setLevel(numeric)
    if fmt != "json":
        for h in root.handlers:
            h.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
