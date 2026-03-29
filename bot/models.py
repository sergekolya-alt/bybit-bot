from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


Side = Literal["LONG", "SHORT"]
SignalSide = Literal["LONG", "SHORT", "NONE"]


@dataclass(frozen=True)
class StrategySignal:
    side: SignalSide
    reason: str
    atr: float


@dataclass(frozen=True)
class FuturesPosition:
    trade_id: int
    symbol: str
    side: Side
    entry_qty: float
    remaining_qty: float
    entry_price: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    risk_per_unit: float
    tp1_hit: bool
    tp2_hit: bool
    opened_at: datetime
