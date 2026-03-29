from __future__ import annotations

from fastapi import FastAPI

from .config import BotConfig
from .health import collect_health_snapshot



def create_health_app(cfg: BotConfig) -> FastAPI:
    app = FastAPI(title="Bybit Bot Health API", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return collect_health_snapshot(cfg)

    @app.get("/live")
    def live() -> dict:
        return {"ok": True}

    return app
