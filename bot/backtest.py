from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any

import pandas as pd

from .ai_filter import AISignalFilter, build_runtime_feature_vector
from .config import BotConfig
from .models import Side
from .strategy import BreakoutMomentumStrategy


@dataclass(frozen=True)
class BacktestTrade:
    side: Side
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    r_multiple: float
    reason: str


class StrategyBacktester:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg
        self.strategy = BreakoutMomentumStrategy(cfg)

    def run(
        self,
        candles_15m: pd.DataFrame,
        candles_1h: pd.DataFrame,
        ai_filter: AISignalFilter | None = None,
    ) -> dict[str, Any]:
        if candles_15m.empty or candles_1h.empty:
            raise ValueError("Backtest requires both 15m and 1h candles")

        c15 = candles_15m.copy().sort_values("ts").reset_index(drop=True)
        c1h = candles_1h.copy().sort_values("ts").reset_index(drop=True)

        min_signal = max(self.cfg.signal_ema_slow, self.cfg.donchian_period, self.cfg.volume_sma_period) + 3
        if len(c15) < min_signal or len(c1h) < self.cfg.trend_ema_slow + 3:
            raise ValueError("Not enough candles for configured indicators")

        equity = self.cfg.starting_capital_usdt
        peak_equity = equity
        max_drawdown = 0.0

        pending_signal = "NONE"
        consecutive_losses = 0
        daily_realized: dict[str, float] = {}
        trades_today: dict[str, int] = {}

        open_pos: dict[str, float | int | str | bool] | None = None
        trades: list[BacktestTrade] = []
        condition_names = ["trend_filter_1h", "adx_filter_1h", "donchian_breakout_15m", "ema_alignment_15m", "volume_filter_15m"]
        long_fail_counts = {name: 0 for name in condition_names}
        short_fail_counts = {name: 0 for name in condition_names}
        evaluated_entry_candles = 0
        ai_filtered_count = 0
        ai_passed_count = 0

        for i in range(min_signal, len(c15)):
            row = c15.iloc[i]
            ts = int(row["ts"])
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            day_key = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()

            hist15 = c15.iloc[: i + 1]
            hist1h = c1h[c1h["ts"] <= ts]
            if len(hist1h) < self.cfg.trend_ema_slow + 3:
                continue

            if open_pos is not None:
                side = open_pos["side"]
                entry_price = float(open_pos["entry_price"])
                remaining_qty = float(open_pos["remaining_qty"])
                stop_loss = float(open_pos["stop_loss"])
                tp1_price = float(open_pos["tp1_price"])
                tp2_price = float(open_pos["tp2_price"])
                risk_per_unit = float(open_pos["risk_per_unit"])

                # Stop has priority in ambiguous candles.
                stop_hit = (low <= stop_loss) if side == "LONG" else (high >= stop_loss)
                if stop_hit:
                    pnl = self._close_piece(side, entry_price, stop_loss, remaining_qty)
                    r_piece = pnl / max(1e-12, risk_per_unit * remaining_qty)
                    open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                    open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                    self._finalize_trade(open_pos, ts, stop_loss, "STOP_LOSS", trades, day_key, daily_realized)
                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    max_drawdown = max(max_drawdown, self._drawdown(peak_equity, equity))
                    consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
                    open_pos = None
                    continue

                # TP1
                if not bool(open_pos["tp1_hit"]):
                    tp1_hit = (high >= tp1_price) if side == "LONG" else (low <= tp1_price)
                    if tp1_hit:
                        qty = float(open_pos["entry_qty"]) * self.cfg.tp1_pct
                        qty = min(qty, float(open_pos["remaining_qty"]))
                        pnl = self._close_piece(side, entry_price, tp1_price, qty)
                        r_piece = pnl / max(1e-12, risk_per_unit * qty)
                        open_pos["remaining_qty"] = float(open_pos["remaining_qty"]) - qty
                        open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                        open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                        open_pos["tp1_hit"] = True
                        equity += pnl

                # TP2
                if not bool(open_pos["tp2_hit"]):
                    tp2_hit = (high >= tp2_price) if side == "LONG" else (low <= tp2_price)
                    if tp2_hit:
                        qty = float(open_pos["entry_qty"]) * self.cfg.tp2_pct
                        qty = min(qty, float(open_pos["remaining_qty"]))
                        pnl = self._close_piece(side, entry_price, tp2_price, qty)
                        r_piece = pnl / max(1e-12, risk_per_unit * qty)
                        open_pos["remaining_qty"] = float(open_pos["remaining_qty"]) - qty
                        open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                        open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                        open_pos["tp2_hit"] = True
                        equity += pnl

                # Trailing stop for remaining 40%
                if bool(open_pos["tp1_hit"]) and bool(open_pos["tp2_hit"]) and float(open_pos["remaining_qty"]) > 0:
                    trailed = self.strategy.trailing_stop(side, hist15, float(open_pos["stop_loss"]))
                    if (side == "LONG" and trailed > float(open_pos["stop_loss"])) or (
                        side == "SHORT" and trailed < float(open_pos["stop_loss"])
                    ):
                        open_pos["stop_loss"] = trailed

                    stop_loss = float(open_pos["stop_loss"])
                    trail_hit = (low <= stop_loss) if side == "LONG" else (high >= stop_loss)
                    if trail_hit:
                        qty = float(open_pos["remaining_qty"])
                        pnl = self._close_piece(side, entry_price, stop_loss, qty)
                        r_piece = pnl / max(1e-12, risk_per_unit * qty)
                        open_pos["remaining_qty"] = 0.0
                        open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                        open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                        self._finalize_trade(open_pos, ts, stop_loss, "TRAIL_STOP", trades, day_key, daily_realized)
                        equity += pnl
                        peak_equity = max(peak_equity, equity)
                        max_drawdown = max(max_drawdown, self._drawdown(peak_equity, equity))
                        consecutive_losses = consecutive_losses + 1 if float(open_pos["realized_pnl"]) < 0 else 0
                        open_pos = None
                        continue

                # Opposite signal closes remainder at candle close.
                if open_pos is not None and self.strategy.is_opposite_signal(side, hist15, hist1h):
                    qty = float(open_pos["remaining_qty"])
                    if qty > 0:
                        pnl = self._close_piece(side, entry_price, close, qty)
                        r_piece = pnl / max(1e-12, risk_per_unit * qty)
                        open_pos["remaining_qty"] = 0.0
                        open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                        open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                        equity += pnl
                    self._finalize_trade(open_pos, ts, close, "OPPOSITE_SIGNAL", trades, day_key, daily_realized)
                    peak_equity = max(peak_equity, equity)
                    max_drawdown = max(max_drawdown, self._drawdown(peak_equity, equity))
                    consecutive_losses = consecutive_losses + 1 if float(open_pos["realized_pnl"]) < 0 else 0
                    open_pos = None
                    continue

                peak_equity = max(peak_equity, equity)
                max_drawdown = max(max_drawdown, self._drawdown(peak_equity, equity))
                continue

            # Flat: entry checks + risk halts.
            daily_pnl = daily_realized.get(day_key, 0.0)
            if daily_pnl <= -abs(equity * self.cfg.daily_loss_limit_pct):
                pending_signal = "NONE"
                continue
            if consecutive_losses >= self.cfg.max_consecutive_losses:
                pending_signal = "NONE"
                continue
            if trades_today.get(day_key, 0) >= self.cfg.max_trades_per_day:
                pending_signal = "NONE"
                continue

            cond = self._entry_condition_flags(hist15, hist1h)
            if cond is not None:
                evaluated_entry_candles += 1
                for name, ok in cond["long"].items():
                    if not ok:
                        long_fail_counts[name] += 1
                for name, ok in cond["short"].items():
                    if not ok:
                        short_fail_counts[name] += 1

            signal_out = self.strategy.entry_signal(hist15, hist1h)
            if signal_out.side == "NONE":
                pending_signal = "NONE"
                continue

            if pending_signal != signal_out.side:
                pending_signal = signal_out.side
                continue

            pending_signal = "NONE"

            if ai_filter is not None and ai_filter.enabled:
                features = build_runtime_feature_vector(hist15, hist1h, self.cfg, signal_out.side)
                if features is None:
                    continue
                decision = ai_filter.decide(features)
                if not decision.allow:
                    ai_filtered_count += 1
                    continue
                ai_passed_count += 1

            entry_side = signal_out.side
            atr_value = signal_out.atr
            if atr_value <= 0:
                continue

            if entry_side == "LONG":
                stop = close - self.cfg.atr_stop_mult * atr_value
            else:
                stop = close + self.cfg.atr_stop_mult * atr_value

            risk_per_unit = abs(close - stop)
            if risk_per_unit <= 0:
                continue

            risk_amount = equity * self.cfg.risk_per_trade_pct
            qty_by_risk = risk_amount / risk_per_unit
            qty_by_lev = (equity * self.cfg.leverage) / close if close > 0 else 0.0
            qty = min(qty_by_risk, qty_by_lev)
            if qty <= 0:
                continue

            if entry_side == "LONG":
                tp1 = close + risk_per_unit * self.cfg.tp1_r
                tp2 = close + risk_per_unit * self.cfg.tp2_r
            else:
                tp1 = close - risk_per_unit * self.cfg.tp1_r
                tp2 = close - risk_per_unit * self.cfg.tp2_r

            open_pos = {
                "side": entry_side,
                "entry_ts": ts,
                "entry_price": close,
                "entry_qty": qty,
                "remaining_qty": qty,
                "stop_loss": stop,
                "tp1_price": tp1,
                "tp2_price": tp2,
                "tp1_hit": False,
                "tp2_hit": False,
                "risk_per_unit": risk_per_unit,
                "realized_pnl": 0.0,
                "realized_r": 0.0,
            }
            trades_today[day_key] = trades_today.get(day_key, 0) + 1

        if open_pos is not None:
            last = c15.iloc[-1]
            final_price = float(last["close"])
            qty = float(open_pos["remaining_qty"])
            if qty > 0:
                pnl = self._close_piece(str(open_pos["side"]), float(open_pos["entry_price"]), final_price, qty)
                r_piece = pnl / max(1e-12, float(open_pos["risk_per_unit"]) * qty)
                open_pos["remaining_qty"] = 0.0
                open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
            day_key = datetime.fromtimestamp(int(last["ts"]) / 1000, tz=timezone.utc).date().isoformat()
            self._finalize_trade(open_pos, int(last["ts"]), final_price, "END_OF_DATA", trades, day_key, daily_realized)

        total_pnl = sum(t.pnl for t in trades)
        trade_count = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = (wins / trade_count * 100.0) if trade_count else 0.0
        avg_r = sum(t.r_multiple for t in trades) / trade_count if trade_count else 0.0
        combined_fail_counts = {
            name: long_fail_counts[name] + short_fail_counts[name]
            for name in condition_names
        }
        dominant_blocker = "none"
        if combined_fail_counts:
            dominant_blocker = max(combined_fail_counts, key=combined_fail_counts.get)

        summary = {
            "total_trades": trade_count,
            "win_rate_pct": round(win_rate, 2),
            "total_pnl": round(total_pnl, 8),
            "max_drawdown_pct": round(max_drawdown * 100.0, 4),
            "average_r_multiple": round(avg_r, 4),
            "ending_equity": round(self.cfg.starting_capital_usdt + total_pnl, 8),
            "entry_diagnostics": {
                "evaluated_entry_candles": evaluated_entry_candles,
                "long_fail_counts": long_fail_counts,
                "short_fail_counts": short_fail_counts,
                "dominant_blocker": dominant_blocker,
            },
        }
        if ai_filter is not None and ai_filter.enabled:
            summary["ai_filter"] = {
                "enabled": True,
                "threshold": ai_filter.threshold,
                "accepted_signals": ai_passed_count,
                "rejected_signals": ai_filtered_count,
            }

        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        return {"summary": summary, "trades": trades_df}

    def _entry_condition_flags(
        self,
        candles_15m: pd.DataFrame,
        candles_1h: pd.DataFrame,
    ) -> dict[str, dict[str, bool]] | None:
        min_signal = max(self.cfg.signal_ema_slow, self.cfg.donchian_period, self.cfg.volume_sma_period) + 3
        min_trend = max(self.cfg.trend_ema_slow, self.cfg.adx_period) + 3
        if len(candles_15m) < min_signal or len(candles_1h) < min_trend:
            return None

        s15 = self.strategy._with_signal_indicators(candles_15m)
        t1h = self.strategy._with_trend_indicators(candles_1h)
        last15 = s15.iloc[-1]
        last1h = t1h.iloc[-1]

        adx_ok = float(last1h["adx"]) > self.cfg.adx_min
        vol_ok = float(last15["volume"]) > float(last15["vol_sma20"]) * 1.0

        return {
            "long": {
                "trend_filter_1h": float(last1h["ema50"]) > float(last1h["ema200"]),
                "adx_filter_1h": adx_ok,
                "donchian_breakout_15m": float(last15["high"]) >= float(last15["donchian_upper"]),
                "ema_alignment_15m": float(last15["ema20"]) > float(last15["ema50"]),
                "volume_filter_15m": vol_ok,
            },
            "short": {
                "trend_filter_1h": float(last1h["ema50"]) < float(last1h["ema200"]),
                "adx_filter_1h": adx_ok,
                "donchian_breakout_15m": float(last15["low"]) <= float(last15["donchian_lower"]),
                "ema_alignment_15m": float(last15["ema20"]) < float(last15["ema50"]),
                "volume_filter_15m": vol_ok,
            },
        }

    @staticmethod
    def _close_piece(side: str, entry_price: float, exit_price: float, qty: float) -> float:
        if side == "LONG":
            return (exit_price - entry_price) * qty
        return (entry_price - exit_price) * qty

    @staticmethod
    def _drawdown(peak: float, equity: float) -> float:
        if peak <= 0:
            return 0.0
        return (peak - equity) / peak

    @staticmethod
    def _finalize_trade(
        open_pos: dict[str, float | int | str | bool],
        exit_ts: int,
        exit_price: float,
        reason: str,
        trades: list[BacktestTrade],
        day_key: str,
        daily_realized: dict[str, float],
    ) -> None:
        pnl = float(open_pos["realized_pnl"])
        daily_realized[day_key] = daily_realized.get(day_key, 0.0) + pnl
        trades.append(
            BacktestTrade(
                side=str(open_pos["side"]),
                entry_ts=int(open_pos["entry_ts"]),
                exit_ts=exit_ts,
                entry_price=float(open_pos["entry_price"]),
                exit_price=exit_price,
                qty=float(open_pos["entry_qty"]),
                pnl=pnl,
                r_multiple=float(open_pos["realized_r"]),
                reason=reason,
            )
        )


def fetch_historical_klines(exchange, symbol: str, interval: str, total_limit: int) -> pd.DataFrame:
    remaining = max(1, total_limit)
    end_ms: int | None = None
    frames: list[pd.DataFrame] = []
    prev_oldest: int | None = None

    while remaining > 0:
        batch = min(1000, remaining)
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = exchange.get_klines(symbol=symbol, interval=interval, limit=batch, end_ms=end_ms)
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(0.4 * (attempt + 1))

        if df.empty:
            break
        frames.append(df)

        oldest = int(df["ts"].min())
        if prev_oldest is not None and oldest >= prev_oldest:
            break
        prev_oldest = oldest
        end_ms = oldest - 1
        if len(df) < batch:
            break
        remaining -= len(df)

    if not frames:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out.tail(total_limit).reset_index(drop=True)


def load_candles_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["ts", "open", "high", "low", "close", "volume"]].copy()
