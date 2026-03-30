from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from .market_structure import RangeDetector, SRDetector, SRZone

if TYPE_CHECKING:
    from .config import BotConfig


@dataclass(frozen=True)
class SignalContext:
    """
    Market structure context computed once per tick and passed to the strategy.

    Contains range bounds, current price location, MR trade permissions,
    and nearest S/R zones. All fields are read-only after construction.
    """

    range_high: float
    range_low: float
    range_mid: float
    price_location: str        # "lower_edge" | "upper_edge" | "middle"
    mr_long_allowed: bool      # True only when price is in lower_edge zone
    mr_short_allowed: bool     # True only when price is in upper_edge zone
    nearest_support: SRZone | None
    nearest_resistance: SRZone | None
    sr_zones: list[SRZone]
    sr_zone_count: int


class MarketContextBuilder:
    """
    Assembles a SignalContext from raw candle data.

    candles_15m is accepted but not used in this implementation — it is
    reserved for future 15m-level S/R detection without changing the call site.
    """

    def __init__(self, cfg: "BotConfig") -> None:
        self.cfg = cfg

    def build(
        self,
        candles_15m: pd.DataFrame,
        candles_1h: pd.DataFrame,
        current_price: float,
    ) -> SignalContext:
        rng = RangeDetector(self.cfg).detect(candles_1h, current_price)
        sr = SRDetector(self.cfg).detect(candles_1h, current_price)

        return SignalContext(
            range_high=rng.range_high,
            range_low=rng.range_low,
            range_mid=rng.range_mid,
            price_location=rng.price_location,
            mr_long_allowed=(rng.price_location == "lower_edge"),
            mr_short_allowed=(rng.price_location == "upper_edge"),
            nearest_support=sr.nearest_support,
            nearest_resistance=sr.nearest_resistance,
            sr_zones=sr.zones,
            sr_zone_count=len(sr.zones),
        )
