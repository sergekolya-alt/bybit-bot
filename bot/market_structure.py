from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import BotConfig


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeResult:
    """Detected price range and edge thresholds for the current market."""

    range_high: float
    range_low: float
    range_mid: float
    upper_edge_threshold: float  # price >= this → upper_edge
    lower_edge_threshold: float  # price <= this → lower_edge
    price_location: str          # "lower_edge" | "upper_edge" | "middle"


@dataclass
class SRZone:
    """A support or resistance zone built from clustered swing levels."""

    zone_low: float
    zone_high: float
    touch_count: int
    zone_type: str  # "support" | "resistance" | "both"
    mid: float = field(init=False)

    def __post_init__(self) -> None:
        self.mid = (self.zone_low + self.zone_high) / 2.0

    def __repr__(self) -> str:
        return (
            f"SRZone({self.zone_type} {self.zone_low:.2f}-{self.zone_high:.2f}"
            f" touches={self.touch_count})"
        )


@dataclass(frozen=True)
class SRResult:
    """All detected S/R zones plus the nearest ones relative to current price."""

    zones: list[SRZone]
    nearest_support: SRZone | None
    nearest_resistance: SRZone | None


# ---------------------------------------------------------------------------
# Range detection
# ---------------------------------------------------------------------------


class RangeDetector:
    """Detects the recent price range and classifies the current price location."""

    def __init__(self, cfg: "BotConfig") -> None:
        self.cfg = cfg

    def detect(self, candles_1h: pd.DataFrame, current_price: float) -> RangeResult:
        """Compute range from the last range_lookback_candles 1h candles."""
        df = candles_1h.tail(self.cfg.range_lookback_candles)
        range_high, range_low = self._compute_range(df)

        span = range_high - range_low
        if span <= 0:
            # Degenerate case: flat candles — treat everything as middle
            return RangeResult(
                range_high=range_high,
                range_low=range_low,
                range_mid=range_high,
                upper_edge_threshold=range_high,
                lower_edge_threshold=range_low,
                price_location="middle",
            )

        range_mid = (range_high + range_low) / 2.0
        upper_edge_threshold = range_high - span * self.cfg.mr_edge_zone_pct
        lower_edge_threshold = range_low + span * self.cfg.mr_edge_zone_pct

        if current_price <= lower_edge_threshold:
            location = "lower_edge"
        elif current_price >= upper_edge_threshold:
            location = "upper_edge"
        else:
            location = "middle"

        return RangeResult(
            range_high=range_high,
            range_low=range_low,
            range_mid=range_mid,
            upper_edge_threshold=upper_edge_threshold,
            lower_edge_threshold=lower_edge_threshold,
            price_location=location,
        )

    def _compute_range(self, df: pd.DataFrame) -> tuple[float, float]:
        """
        Override this method to use a different range detection algorithm
        (e.g. ATR-based, Donchian, or multi-timeframe).
        """
        return float(df["high"].max()), float(df["low"].min())


# ---------------------------------------------------------------------------
# Swing high / low detection
# ---------------------------------------------------------------------------


class SwingDetector:
    """Detects swing highs and lows using a rolling window comparison."""

    def __init__(self, window: int) -> None:
        self.window = window

    def find_swings(
        self, candles: pd.DataFrame
    ) -> tuple[list[float], list[float]]:
        """
        Return (swing_highs, swing_lows) as lists of price levels.

        A candle at index i is a swing high if its high is strictly greater
        than the high of every candle in [i-window, i) and (i, i+window].
        Same logic applies for swing lows.
        """
        window = self.window
        highs = candles["high"].to_numpy(dtype=float)
        lows = candles["low"].to_numpy(dtype=float)
        n = len(highs)

        swing_highs: list[float] = []
        swing_lows: list[float] = []

        for i in range(window, n - window):
            left_h = highs[i - window: i]
            right_h = highs[i + 1: i + window + 1]
            if highs[i] > left_h.max() and highs[i] > right_h.max():
                swing_highs.append(float(highs[i]))

            left_l = lows[i - window: i]
            right_l = lows[i + 1: i + window + 1]
            if lows[i] < left_l.min() and lows[i] < right_l.min():
                swing_lows.append(float(lows[i]))

        return swing_highs, swing_lows


# ---------------------------------------------------------------------------
# Support / Resistance detection
# ---------------------------------------------------------------------------


class SRDetector:
    """Detects support and resistance zones from swing levels."""

    def __init__(self, cfg: "BotConfig") -> None:
        self.cfg = cfg
        self._swing_detector = SwingDetector(cfg.sr_swing_window)

    def detect(self, candles: pd.DataFrame, current_price: float) -> SRResult:
        """
        Build SR zones from swing highs/lows in the provided candle history.
        Returns nearest support and resistance relative to current_price.
        """
        min_candles_needed = self.cfg.sr_swing_window * 2 + 1
        if len(candles) < min_candles_needed:
            return SRResult(zones=[], nearest_support=None, nearest_resistance=None)

        swing_highs, swing_lows = self._swing_detector.find_swings(candles)
        all_levels = swing_highs + swing_lows

        if not all_levels:
            return SRResult(zones=[], nearest_support=None, nearest_resistance=None)

        # Cluster nearby levels into zones
        raw_zones = self._cluster_levels(all_levels, self.cfg.sr_cluster_tolerance_pct)

        # Filter by minimum touches and assign zone type
        zones: list[SRZone] = []
        for zone in raw_zones:
            if zone.touch_count < self.cfg.sr_min_touches:
                continue
            zone.zone_type = self._classify_zone(zone, current_price)
            zones.append(zone)

        # Sort by strength (touch count desc), keep top N
        zones.sort(key=lambda z: z.touch_count, reverse=True)
        zones = zones[: self.cfg.sr_max_levels]

        nearest_support = self._nearest_support(zones, current_price)
        nearest_resistance = self._nearest_resistance(zones, current_price)

        return SRResult(
            zones=zones,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cluster_levels(
        self, levels: list[float], tolerance_pct: float
    ) -> list[SRZone]:
        """
        Greedy merge: sort levels ascending, merge into a cluster if the
        new level is within tolerance_pct of the current cluster midpoint.
        """
        if not levels:
            return []

        sorted_levels = sorted(levels)
        zones: list[SRZone] = []
        cluster: list[float] = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            cluster_mid = sum(cluster) / len(cluster)
            if cluster_mid > 0 and abs(level - cluster_mid) / cluster_mid <= tolerance_pct:
                cluster.append(level)
            else:
                zones.append(
                    SRZone(
                        zone_low=min(cluster),
                        zone_high=max(cluster),
                        touch_count=len(cluster),
                        zone_type="both",  # classified later
                    )
                )
                cluster = [level]

        zones.append(
            SRZone(
                zone_low=min(cluster),
                zone_high=max(cluster),
                touch_count=len(cluster),
                zone_type="both",
            )
        )
        return zones

    @staticmethod
    def _classify_zone(zone: SRZone, current_price: float) -> str:
        """
        Classify zone as support, resistance, or both based on position
        relative to current price.
        """
        if zone.zone_high < current_price:
            return "support"
        if zone.zone_low > current_price:
            return "resistance"
        return "both"  # zone straddles current price

    @staticmethod
    def _nearest_support(zones: list[SRZone], current_price: float) -> SRZone | None:
        """Highest zone whose zone_high is below current price."""
        candidates = [z for z in zones if z.zone_high < current_price]
        if not candidates:
            return None
        return max(candidates, key=lambda z: z.zone_high)

    @staticmethod
    def _nearest_resistance(zones: list[SRZone], current_price: float) -> SRZone | None:
        """Lowest zone whose zone_low is above current price."""
        candidates = [z for z in zones if z.zone_low > current_price]
        if not candidates:
            return None
        return min(candidates, key=lambda z: z.zone_low)
