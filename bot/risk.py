from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import BotConfig
from .db import Database


@dataclass(frozen=True)
class RiskGate:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self, cfg: BotConfig, db: Database) -> None:
        self.cfg = cfg
        self.db = db

    @staticmethod
    def utc_day_start(now: datetime | None = None) -> datetime:
        now = now or datetime.utcnow()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def utc_week_start(now: datetime | None = None) -> datetime:
        now = now or datetime.utcnow()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start - timedelta(days=day_start.weekday())

    def check_new_trade_allowed(self, equity_usdt: float) -> RiskGate:
        day_start = self.utc_day_start()
        week_start = self.utc_week_start()

        daily_pnl = self.db.get_daily_realized_pnl(day_start)
        weekly_pnl = self.db.get_weekly_realized_pnl(week_start)

        if daily_pnl <= -abs(equity_usdt * self.cfg.daily_loss_limit_pct):
            return RiskGate(False, "daily_loss_limit_reached")

        if weekly_pnl <= -abs(equity_usdt * self.cfg.weekly_loss_limit_pct):
            return RiskGate(False, "weekly_loss_limit_reached")

        losses = self.db.get_consecutive_losses(self.cfg.symbol)
        if losses >= self.cfg.max_consecutive_losses:
            return RiskGate(False, "max_consecutive_losses_reached")

        trades_today = self.db.get_trade_count_today(self.cfg.symbol, day_start)
        if trades_today >= self.cfg.max_trades_per_day:
            return RiskGate(False, "max_trades_per_day_reached")

        return RiskGate(True, "ok")

    def calculate_position_size(
        self,
        equity_usdt: float,
        entry_price: float,
        stop_loss: float,
        min_qty: float,
        qty_step: float,
        min_notional: float,
    ) -> float:
        if entry_price <= 0:
            return 0.0

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return 0.0

        risk_amount = equity_usdt * self.cfg.risk_per_trade_pct
        qty_by_risk = risk_amount / stop_distance

        max_notional = equity_usdt * self.cfg.leverage
        qty_by_leverage = max_notional / entry_price
        qty = min(qty_by_risk, qty_by_leverage)

        if qty_step > 0:
            qty = int(qty / qty_step) * qty_step

        if qty < min_qty:
            return 0.0

        if min_notional > 0 and qty * entry_price < min_notional:
            return 0.0

        return max(0.0, qty)
