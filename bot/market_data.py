from __future__ import annotations

import logging

import pandas as pd

from .config import BotConfig
from .db import Database
from .exchange import BybitSpotClient


class CandleDataService:
    def __init__(self, cfg: BotConfig, db: Database, exchange: BybitSpotClient) -> None:
        self.cfg = cfg
        self.db = db
        self.exchange = exchange
        self._logger = logging.getLogger(__name__)

    def warmup_cache(self) -> pd.DataFrame:
        return self.fetch_history_and_cache(self.cfg.candle_cache_warmup)

    def get_recent_candles(self, limit: int) -> pd.DataFrame:
        refresh_size = min(1000, max(200, limit))
        latest = self.exchange.get_klines(
            symbol=self.cfg.symbol,
            interval=self.cfg.timeframe,
            limit=refresh_size,
        )
        self.db.upsert_candles(self.cfg.symbol, self.cfg.timeframe, latest)

        cached = self.db.get_cached_candles(self.cfg.symbol, self.cfg.timeframe, limit)
        if len(cached) >= limit:
            return cached

        self._logger.info(
            "Cache has insufficient candles, fetching historical",
            extra={"ctx": {"cached": len(cached), "required": limit}},
        )
        self.fetch_history_and_cache(max(limit, self.cfg.history_fetch_limit))
        return self.db.get_cached_candles(self.cfg.symbol, self.cfg.timeframe, limit)

    def fetch_history_and_cache(self, total_limit: int) -> pd.DataFrame:
        remaining = max(total_limit, 1)
        frames: list[pd.DataFrame] = []
        end_ms: int | None = None
        last_oldest: int | None = None

        while remaining > 0:
            batch_limit = min(1000, remaining)
            df = self.exchange.get_klines(
                symbol=self.cfg.symbol,
                interval=self.cfg.timeframe,
                limit=batch_limit,
                end_ms=end_ms,
            )
            if df.empty:
                break

            self.db.upsert_candles(self.cfg.symbol, self.cfg.timeframe, df)
            frames.append(df)

            oldest = int(df["ts"].min())
            if last_oldest is not None and oldest >= last_oldest:
                break
            last_oldest = oldest
            end_ms = oldest - 1

            if len(df) < batch_limit:
                break
            remaining -= len(df)

        if not frames:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        return merged.tail(total_limit).reset_index(drop=True)
