from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv



def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BotConfig:
    api_key: str
    api_secret: str
    bybit_testnet: bool
    dry_run: bool
    live_trade_confirmation: bool

    symbol: str
    poll_interval_sec: int
    signal_interval: str
    trend_interval: str
    signal_candle_limit: int
    trend_candle_limit: int

    leverage: int
    risk_per_trade_pct: float
    daily_loss_limit_pct: float
    weekly_loss_limit_pct: float
    max_consecutive_losses: int
    max_trades_per_day: int

    signal_ema_fast: int
    signal_ema_slow: int
    atr_period: int
    donchian_period: int
    volume_sma_period: int

    trend_ema_fast: int
    trend_ema_slow: int
    adx_period: int
    adx_min: float
    volume_spike_mult: float

    atr_stop_mult: float
    tp1_r: float
    tp2_r: float
    tp1_pct: float
    tp2_pct: float

    starting_capital_usdt: float
    db_path: str

    telegram_bot_token: str
    telegram_chat_id: str

    log_level: str
    backtest_trades_csv: str
    ai_filter_enabled: bool
    ai_model_path: str
    ai_score_threshold: float

    @staticmethod
    def from_env() -> "BotConfig":
        load_dotenv()

        cfg = BotConfig(
            api_key=os.getenv("BYBIT_API_KEY", "").strip(),
            api_secret=os.getenv("BYBIT_API_SECRET", "").strip(),
            bybit_testnet=_env_bool("BYBIT_TESTNET", True),
            dry_run=_env_bool("DRY_RUN", True),
            live_trade_confirmation=_env_bool("LIVE_TRADE_CONFIRMATION", False),
            symbol=os.getenv("SYMBOL", "BTCUSDT").strip().upper(),
            poll_interval_sec=int(os.getenv("POLL_INTERVAL_SEC", "30")),
            signal_interval=os.getenv("SIGNAL_INTERVAL", "15").strip(),
            trend_interval=os.getenv("TREND_INTERVAL", "60").strip(),
            signal_candle_limit=int(os.getenv("SIGNAL_CANDLE_LIMIT", "300")),
            trend_candle_limit=int(os.getenv("TREND_CANDLE_LIMIT", "300")),
            leverage=int(os.getenv("LEVERAGE", "5")),
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.015")),
            daily_loss_limit_pct=float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.04")),
            weekly_loss_limit_pct=float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", "0.08")),
            max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")),
            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "5")),
            signal_ema_fast=int(os.getenv("SIGNAL_EMA_FAST", "20")),
            signal_ema_slow=int(os.getenv("SIGNAL_EMA_SLOW", "50")),
            atr_period=int(os.getenv("ATR_PERIOD", "14")),
            donchian_period=int(os.getenv("DONCHIAN_PERIOD", "20")),
            volume_sma_period=int(os.getenv("VOLUME_SMA_PERIOD", "20")),
            trend_ema_fast=int(os.getenv("TREND_EMA_FAST", "50")),
            trend_ema_slow=int(os.getenv("TREND_EMA_SLOW", "200")),
            adx_period=int(os.getenv("ADX_PERIOD", "14")),
            adx_min=float(os.getenv("ADX_MIN", "18")),
            volume_spike_mult=float(os.getenv("VOLUME_SPIKE_MULT", "1.5")),
            atr_stop_mult=float(os.getenv("ATR_STOP_MULT", "1.8")),
            tp1_r=float(os.getenv("TP1_R", "1.5")),
            tp2_r=float(os.getenv("TP2_R", "3.0")),
            tp1_pct=float(os.getenv("TP1_PCT", "0.30")),
            tp2_pct=float(os.getenv("TP2_PCT", "0.30")),
            starting_capital_usdt=float(os.getenv("STARTING_CAPITAL_USDT", "1000")),
            db_path=os.getenv("DB_PATH", "bot.sqlite3").strip(),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            backtest_trades_csv=os.getenv("BACKTEST_TRADES_CSV", "backtest_trades.csv").strip(),
            ai_filter_enabled=_env_bool("AI_FILTER_ENABLED", False),
            ai_model_path=os.getenv("AI_MODEL_PATH", "models/ai_signal_filter.json").strip(),
            ai_score_threshold=float(os.getenv("AI_SCORE_THRESHOLD", "0.50")),
        )

        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.symbol != "BTCUSDT":
            raise ValueError("This bot supports BTCUSDT only.")

        if self.signal_interval != "15" or self.trend_interval != "60":
            raise ValueError("This strategy requires SIGNAL_INTERVAL=15 and TREND_INTERVAL=60.")

        if self.poll_interval_sec <= 0:
            raise ValueError("POLL_INTERVAL_SEC must be positive.")

        if self.signal_candle_limit < 120 or self.trend_candle_limit < 220:
            raise ValueError("Candle limits are too low for indicators.")

        if self.signal_ema_fast >= self.signal_ema_slow:
            raise ValueError("SIGNAL_EMA_FAST must be less than SIGNAL_EMA_SLOW.")

        if self.trend_ema_fast >= self.trend_ema_slow:
            raise ValueError("TREND_EMA_FAST must be less than TREND_EMA_SLOW.")

        if self.leverage < 1 or self.leverage > 25:
            raise ValueError("LEVERAGE must be between 1 and 25.")

        if self.risk_per_trade_pct <= 0 or self.risk_per_trade_pct > 0.05:
            raise ValueError("RISK_PER_TRADE_PCT must be in (0, 0.05].")

        if self.daily_loss_limit_pct <= 0 or self.daily_loss_limit_pct > 0.2:
            raise ValueError("DAILY_LOSS_LIMIT_PCT must be in (0, 0.2].")

        if self.weekly_loss_limit_pct <= 0 or self.weekly_loss_limit_pct > 0.3:
            raise ValueError("WEEKLY_LOSS_LIMIT_PCT must be in (0, 0.3].")

        if self.max_consecutive_losses < 1:
            raise ValueError("MAX_CONSECUTIVE_LOSSES must be >= 1.")

        if self.max_trades_per_day < 1:
            raise ValueError("MAX_TRADES_PER_DAY must be >= 1.")

        if self.tp1_pct <= 0 or self.tp2_pct <= 0 or (self.tp1_pct + self.tp2_pct) >= 1:
            raise ValueError("TP percentages must be positive and leave remainder for trailing leg.")

        if self.starting_capital_usdt <= 0:
            raise ValueError("STARTING_CAPITAL_USDT must be positive.")

        if self.ai_score_threshold < 0 or self.ai_score_threshold > 1:
            raise ValueError("AI_SCORE_THRESHOLD must be in [0, 1].")

        if not self.dry_run:
            if not self.api_key or not self.api_secret:
                raise ValueError("Live mode requires BYBIT_API_KEY and BYBIT_API_SECRET.")
            if not self.live_trade_confirmation:
                raise ValueError("Set LIVE_TRADE_CONFIRMATION=true to allow live orders.")
