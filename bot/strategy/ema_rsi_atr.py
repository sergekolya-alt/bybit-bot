from __future__ import annotations

import pandas as pd

from ..config import BotConfig
from ..models import Side, StrategySignal
from .indicators import adx, atr, donchian_lower, donchian_upper, ema, sma


class BreakoutMomentumStrategy:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg

    def _with_signal_indicators(self, candles_15m: pd.DataFrame) -> pd.DataFrame:
        df = candles_15m.copy()
        df["ema20"] = ema(df["close"], self.cfg.signal_ema_fast)
        df["ema50"] = ema(df["close"], self.cfg.signal_ema_slow)
        df["atr"] = atr(df["high"], df["low"], df["close"], self.cfg.atr_period)
        df["donchian_upper"] = donchian_upper(df["high"], self.cfg.donchian_period)
        df["donchian_lower"] = donchian_lower(df["low"], self.cfg.donchian_period)
        df["vol_sma20"] = sma(df["volume"], self.cfg.volume_sma_period)
        return df

    def _with_trend_indicators(self, candles_1h: pd.DataFrame) -> pd.DataFrame:
        df = candles_1h.copy()
        df["ema50"] = ema(df["close"], self.cfg.trend_ema_fast)
        df["ema200"] = ema(df["close"], self.cfg.trend_ema_slow)
        df["adx"] = adx(df["high"], df["low"], df["close"], self.cfg.adx_period)
        return df

    def entry_signal(self, candles_15m: pd.DataFrame, candles_1h: pd.DataFrame) -> StrategySignal:
        min_signal = max(self.cfg.signal_ema_slow, self.cfg.donchian_period, self.cfg.volume_sma_period) + 3
        min_trend = max(self.cfg.trend_ema_slow, self.cfg.adx_period) + 3
        if len(candles_15m) < min_signal or len(candles_1h) < min_trend:
            return StrategySignal(side="NONE", reason="not_enough_data", atr=0.0)

        s15 = self._with_signal_indicators(candles_15m)
        t1h = self._with_trend_indicators(candles_1h)

        last15 = s15.iloc[-1]
        last1h = t1h.iloc[-1]

        adx_value = float(last1h["adx"])
        in_breakout_regime = adx_value > self.cfg.adx_min
        in_mean_reversion_regime = adx_value < self.cfg.mean_reversion_adx_max

        ema20_15 = float(last15["ema20"])
        ema50_15 = float(last15["ema50"])
        ema50_1h = float(last1h["ema50"])
        ema200_1h = float(last1h["ema200"])
        atr_value = float(last15["atr"])

        # Donchian with 0.5% tolerance — near-breakout also counts
        donchian_upper_val = float(last15["donchian_upper"])
        donchian_lower_val = float(last15["donchian_lower"])
        long_breakout = float(last15["high"]) >= donchian_upper_val * 0.995
        short_breakout = float(last15["low"]) <= donchian_lower_val * 1.005

        # --- Breakout regime (priority) ---
        long_conditions = [
            ema50_1h > ema200_1h,
            in_breakout_regime,
            long_breakout,
            ema20_15 > ema50_15,
        ]
        short_conditions = [
            ema50_1h < ema200_1h,
            in_breakout_regime,
            short_breakout,
            ema20_15 < ema50_15,
        ]

        if all(long_conditions) and atr_value > 0:
            return StrategySignal(side="LONG", reason="breakout_long", atr=atr_value)

        if all(short_conditions) and atr_value > 0:
            return StrategySignal(side="SHORT", reason="breakout_short", atr=atr_value)

        # --- Mean reversion regime (ranging market) ---
        if in_mean_reversion_regime and atr_value > 0:
            mr_long_conditions = [
                ema50_1h > ema200_1h,                           # 1h uptrend
                float(last15["low"]) < ema20_15,                # pulled back below 15m EMA20
                float(last15["close"]) > ema50_15,              # still above 15m EMA50 (support holds)
                float(last15["close"]) > float(last15["open"]), # bullish candle (bounce forming)
            ]
            mr_short_conditions = [
                ema50_1h < ema200_1h,                           # 1h downtrend
                float(last15["high"]) > ema20_15,               # bounced above 15m EMA20
                float(last15["close"]) < ema50_15,              # still below 15m EMA50 (resistance holds)
                float(last15["close"]) < float(last15["open"]), # bearish candle (rejection forming)
            ]

            if all(mr_long_conditions):
                return StrategySignal(side="LONG", reason="mean_reversion_long", atr=atr_value)

            if all(mr_short_conditions):
                return StrategySignal(side="SHORT", reason="mean_reversion_short", atr=atr_value)

        return StrategySignal(side="NONE", reason="no_signal", atr=atr_value if atr_value > 0 else 0.0)

    def is_opposite_signal(self, current_side: Side, candles_15m: pd.DataFrame, candles_1h: pd.DataFrame) -> bool:
        signal = self.entry_signal(candles_15m, candles_1h)
        if current_side == "LONG":
            return signal.side == "SHORT"
        return signal.side == "LONG"

    @staticmethod
    def trailing_stop(side: Side, candles_15m: pd.DataFrame, current_stop: float) -> float:
        if len(candles_15m) < 4:
            return current_stop

        recent = candles_15m.iloc[-4:-1]
        if side == "LONG":
            swing = float(recent["low"].min())
            return max(current_stop, swing)

        swing = float(recent["high"].max())
        return min(current_stop, swing)
