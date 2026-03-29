from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class StructuredLogFormatter(logging.Formatter):
    """Simple JSON formatter for machine-friendly logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict) and ctx:
            payload["ctx"] = ctx

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        base = f"{ts} | {record.levelname} | {record.name} | {record.getMessage()}"
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict) and ctx:
            return f"{base} | ctx={ctx}"
        return base


def setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredLogFormatter() if json_logs else PlainFormatter())

    root.handlers.clear()
    root.addHandler(handler)
