from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StatusStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def load(self) -> dict[str, Any]:
        with self._lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "state": "unknown",
                "updated_at": self._now_iso(),
            }
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {
                "state": "unknown",
                "updated_at": self._now_iso(),
            }

    def update(self, **fields: Any) -> None:
        with self._lock:
            current = self._read_unlocked() if self.path.exists() else {}
            current.update(fields)
            current["updated_at"] = self._now_iso()

            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(current, ensure_ascii=True, indent=2), encoding="utf-8")
            tmp.replace(self.path)
