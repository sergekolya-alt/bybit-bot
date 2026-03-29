from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bot.ai_filter import (
    AISignalFilter,
    build_runtime_feature_vector,
    build_signal_dataset,
    train_ai_model,
    train_rf_model,
)
from bot.backtest import StrategyBacktester, fetch_historical_klines, load_candles_csv
from bot.config import BotConfig
from bot.db import Database
from bot.engine import TradingBot
from bot.exchange import BybitClient
from bot.strategy import BreakoutMomentumStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTCUSDT futures trading bot")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run-bot", help="Run the live/dry-run bot")
    p_run.add_argument("--once", action="store_true", help="Run one cycle and exit")
    p_run.add_argument(
        "--paper",
        action="store_true",
        help="Run paper/live-shadow mode (no real orders, RF filter forced at 0.65)",
    )

    p_bt = sub.add_parser("backtest", help="Run backtest with the same strategy")
    p_bt.add_argument("--limit", type=int, default=3000, help="Candle count for backtest")
    p_bt.add_argument("--candles15-csv", type=str, default="", help="Optional 15m candles CSV")
    p_bt.add_argument("--candles1h-csv", type=str, default="", help="Optional 1h candles CSV")
    p_bt.add_argument("--output", type=str, default="", help="Backtest trades CSV output path")
    p_bt.add_argument("--use-ai-filter", action="store_true", help="Apply AI signal-quality filter")
    p_bt.add_argument(
        "--compare-ai-filter",
        action="store_true",
        help="Run baseline and AI-filtered backtests side-by-side",
    )

    p_ds = sub.add_parser("generate-dataset", help="Generate ML dataset from candidate signals")
    p_ds.add_argument("--limit", type=int, default=3000, help="Candle count for dataset generation")
    p_ds.add_argument("--output", type=str, default="ai_signal_dataset.csv", help="Output CSV path")
    p_ds.add_argument("--candles15-csv", type=str, default="", help="Optional 15m candles CSV")
    p_ds.add_argument("--candles1h-csv", type=str, default="", help="Optional 1h candles CSV")

    p_train = sub.add_parser("train-ai", help="Train AI signal-quality model")
    p_train.add_argument("--dataset", type=str, default="ai_signal_dataset.csv", help="Input dataset CSV")
    p_train.add_argument(
        "--model-output",
        type=str,
        default="models/ai_signal_filter.json",
        help="Output model artifact path",
    )

    p_train_model = sub.add_parser("train-model", help="Train RandomForest model for paper/live-shadow mode")
    p_train_model.add_argument("--dataset", type=str, default="ai_signal_dataset.csv", help="Input dataset CSV")
    p_train_model.add_argument(
        "--output",
        type=str,
        default="models/rf_signal_filter.joblib",
        help="Output model artifact path",
    )

    sub.add_parser("paper-report", help="Print paper trading activity summary")

    p_diag = sub.add_parser("diagnose-runtime", help="Diagnose runtime gate flow on historical candles")
    p_diag.add_argument("--window", type=int, default=10000, help="15m candle window size")
    p_diag.add_argument("--candles15-csv", type=str, default="", help="Optional 15m candles CSV")
    p_diag.add_argument("--candles1h-csv", type=str, default="", help="Optional 1h candles CSV")
    p_diag.add_argument(
        "--ai-thresholds",
        type=str,
        default="",
        help="Optional comma-separated AI thresholds for sensitivity run (example: 0.55,0.60,0.65)",
    )

    return parser.parse_args()


def _load_backtest_candles(
    cfg: BotConfig,
    limit: int,
    candles15_csv: str,
    candles1h_csv: str,
) -> tuple:
    required_1h = max(limit // 4, cfg.trend_candle_limit)

    if bool(candles15_csv) != bool(candles1h_csv):
        raise ValueError("Provide both --candles15-csv and --candles1h-csv, or neither.")

    if candles15_csv and candles1h_csv:
        candles_15m = load_candles_csv(candles15_csv)
        candles_1h = load_candles_csv(candles1h_csv)
        return candles_15m, candles_1h, required_1h

    def _merge_candles(primary, secondary, take_limit: int):
        if primary is None or len(primary) == 0:
            out = secondary.copy()
        elif secondary is None or len(secondary) == 0:
            out = primary.copy()
        else:
            out = (
                pd.concat([primary, secondary], ignore_index=True)
                .drop_duplicates(subset=["ts"])
                .sort_values("ts")
                .reset_index(drop=True)
            )
        return out.tail(take_limit).reset_index(drop=True)

    exchange = BybitClient(cfg.api_key, cfg.api_secret, cfg.bybit_testnet)
    db = Database(cfg.db_path)
    try:
        cached_15m = db.get_cached_candles(cfg.symbol, cfg.signal_interval, limit)
        cached_1h = db.get_cached_candles(cfg.symbol, cfg.trend_interval, required_1h)

        fetched_15m = cached_15m.iloc[0:0]
        fetched_1h = cached_1h.iloc[0:0]

        try:
            fetched_15m = fetch_historical_klines(
                exchange=exchange,
                symbol=cfg.symbol,
                interval=cfg.signal_interval,
                total_limit=limit,
            )
            if not fetched_15m.empty:
                db.upsert_candles(cfg.symbol, cfg.signal_interval, fetched_15m)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Backtest 15m fetch failed, trying local cache: %s", exc)

        try:
            fetched_1h = fetch_historical_klines(
                exchange=exchange,
                symbol=cfg.symbol,
                interval=cfg.trend_interval,
                total_limit=required_1h,
            )
            if not fetched_1h.empty:
                db.upsert_candles(cfg.symbol, cfg.trend_interval, fetched_1h)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Backtest 1h fetch failed, trying local cache: %s", exc)

        candles_15m = _merge_candles(fetched_15m, cached_15m, limit)
        candles_1h = _merge_candles(fetched_1h, cached_1h, required_1h)

    finally:
        db.close()

    return candles_15m, candles_1h, required_1h


def run_bot(cfg: BotConfig, once: bool, paper: bool = False) -> None:
    if paper:
        paper_model_path = cfg.ai_model_path
        if paper_model_path == "models/ai_signal_filter.json":
            paper_model_path = "models/rf_signal_filter.joblib"
        cfg = replace(
            cfg,
            dry_run=True,
            ai_filter_enabled=True,
            ai_score_threshold=0.65,
            ai_model_path=paper_model_path,
        )
        model_path = Path(cfg.ai_model_path)
        if not model_path.exists():
            raise ValueError(
                (
                    "Paper mode requires a trained AI model file. "
                    f"Not found: {cfg.ai_model_path}. "
                    "Set AI_MODEL_PATH to your trained RandomForest .joblib file if needed."
                )
            )

    bot = TradingBot(cfg, paper_mode=paper)
    bot.run(once=once)


def run_backtest(
    cfg: BotConfig,
    limit: int,
    candles15_csv: str,
    candles1h_csv: str,
    output: str,
    use_ai_filter: bool,
    compare_ai_filter: bool,
) -> None:
    candles_15m, candles_1h, required_1h = _load_backtest_candles(cfg, limit, candles15_csv, candles1h_csv)

    if len(candles_15m) < limit or len(candles_1h) < required_1h:
        raise ValueError(
            (
                "Insufficient historical data for requested backtest limit: "
                f"need 15m={limit} and 1h={required_1h}, got 15m={len(candles_15m)} and 1h={len(candles_1h)}"
            )
        )

    backtester = StrategyBacktester(cfg)
    output_path = output or cfg.backtest_trades_csv

    if compare_ai_filter:
        baseline = backtester.run(candles_15m, candles_1h)
        baseline_csv = output_path.replace(".csv", "_baseline.csv")
        baseline["trades"].to_csv(baseline_csv, index=False)

        ai_filter = AISignalFilter(True, cfg.ai_model_path, cfg.ai_score_threshold)
        with_ai = backtester.run(candles_15m, candles_1h, ai_filter=ai_filter)
        with_ai_csv = output_path.replace(".csv", "_ai.csv")
        with_ai["trades"].to_csv(with_ai_csv, index=False)

        print(
            json.dumps(
                {
                    "baseline": baseline["summary"],
                    "ai_filter": with_ai["summary"],
                    "baseline_trades_csv": baseline_csv,
                    "ai_trades_csv": with_ai_csv,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return

    ai_filter = None
    if use_ai_filter:
        ai_filter = AISignalFilter(True, cfg.ai_model_path, cfg.ai_score_threshold)

    result = backtester.run(candles_15m, candles_1h, ai_filter=ai_filter)
    trades_df = result["trades"]
    trades_df.to_csv(output_path, index=False)

    print(
        json.dumps(
            {
                "summary": result["summary"],
                "trades_csv": output_path,
                "trades_count": len(trades_df),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


def run_generate_dataset(
    cfg: BotConfig,
    limit: int,
    output: str,
    candles15_csv: str,
    candles1h_csv: str,
) -> None:
    candles_15m, candles_1h, required_1h = _load_backtest_candles(cfg, limit, candles15_csv, candles1h_csv)

    if len(candles_15m) < limit or len(candles_1h) < required_1h:
        raise ValueError(
            (
                "Insufficient historical data for dataset generation: "
                f"need 15m={limit} and 1h={required_1h}, got 15m={len(candles_15m)} and 1h={len(candles_1h)}"
            )
        )

    ds, stats = build_signal_dataset(candles_15m, candles_1h, cfg, return_stats=True)
    ds.to_csv(output, index=False)

    print(
        json.dumps(
            {
                "dataset_csv": output,
                "candles_15m_loaded": int(len(candles_15m)),
                "candles_1h_loaded": int(len(candles_1h)),
                "candidate_signals": int(stats["candidate_signals"]),
                "dataset_rows": int(len(ds)),
                "positive_rate": float(ds["target"].mean()) if len(ds) else 0.0,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


def run_train_ai(dataset: str, model_output: str) -> None:
    metrics = train_ai_model(dataset, model_output)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


def run_train_model(dataset: str, output: str) -> None:
    metrics = train_rf_model(dataset, output)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


def run_paper_report(cfg: BotConfig) -> None:
    db = Database(cfg.db_path)
    try:
        report = db.get_paper_report(cfg.starting_capital_usdt)
    finally:
        db.close()
    print(json.dumps(report, indent=2, ensure_ascii=True))


def run_diagnose_runtime(
    cfg: BotConfig,
    window: int,
    candles15_csv: str,
    candles1h_csv: str,
    emit_json: bool = True,
) -> dict:
    candles_15m, candles_1h, required_1h = _load_backtest_candles(cfg, window, candles15_csv, candles1h_csv)
    if len(candles_15m) < window or len(candles_1h) < required_1h:
        raise ValueError(
            (
                "Insufficient historical data for diagnose-runtime: "
                f"need 15m={window} and 1h={required_1h}, got 15m={len(candles_15m)} and 1h={len(candles_1h)}"
            )
        )

    c15 = candles_15m.copy().sort_values("ts").reset_index(drop=True)
    c1h = candles_1h.copy().sort_values("ts").reset_index(drop=True)
    strategy = BreakoutMomentumStrategy(cfg)
    ai_filter = AISignalFilter(cfg.ai_filter_enabled, cfg.ai_model_path, cfg.ai_score_threshold)

    min_signal = max(cfg.signal_ema_slow, cfg.donchian_period, cfg.volume_sma_period) + 3
    min_trend = max(cfg.trend_ema_slow, cfg.adx_period) + 3

    equity = cfg.starting_capital_usdt
    pending_signal = "NONE"
    consecutive_losses = 0
    daily_realized: dict[str, float] = {}
    weekly_realized: dict[str, float] = {}
    trades_today: dict[str, int] = {}
    open_pos: dict[str, float | int | str | bool] | None = None
    evaluated_entry_candles = 0

    def _bucket() -> dict[str, int]:
        return {"total": 0, "long": 0, "short": 0}

    counts = {
        "raw_strategy_signals": _bucket(),
        "confirmed_signals": _bucket(),
        "ai_accepted_signals": _bucket(),
        "ai_rejected_signals": _bucket(),
        "risk_allowed_signals": _bucket(),
        "risk_blocked_signals": _bucket(),
        "trades_opened": _bucket(),
    }
    risk_gate = {"allowed_checks": 0, "blocked_checks": 0, "blocked_reasons": {}}

    def _add(bucket: dict[str, int], side: str) -> None:
        bucket["total"] += 1
        if side == "LONG":
            bucket["long"] += 1
        elif side == "SHORT":
            bucket["short"] += 1

    def _day_key(ts_ms: int) -> str:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()

    def _week_key(ts_ms: int) -> str:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
        start = dt.fromordinal(dt.toordinal() - dt.weekday())
        return start.isoformat()

    def _close_piece(side: str, entry_price: float, exit_price: float, qty: float) -> float:
        if side == "LONG":
            return (exit_price - entry_price) * qty
        return (entry_price - exit_price) * qty

    for i in range(min_signal, len(c15)):
        row = c15.iloc[i]
        ts = int(row["ts"])
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        day_key = _day_key(ts)
        week_key = _week_key(ts)

        hist15 = c15.iloc[: i + 1]
        hist1h = c1h[c1h["ts"] <= ts]
        if len(hist1h) < min_trend:
            continue

        if open_pos is not None:
            side = str(open_pos["side"])
            entry_price = float(open_pos["entry_price"])
            remaining_qty = float(open_pos["remaining_qty"])
            stop_loss = float(open_pos["stop_loss"])
            tp1_price = float(open_pos["tp1_price"])
            tp2_price = float(open_pos["tp2_price"])
            risk_per_unit = float(open_pos["risk_per_unit"])

            stop_hit = (low <= stop_loss) if side == "LONG" else (high >= stop_loss)
            if stop_hit:
                pnl = _close_piece(side, entry_price, stop_loss, remaining_qty)
                r_piece = pnl / max(1e-12, risk_per_unit * remaining_qty)
                open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                total_pnl = float(open_pos["realized_pnl"])
                equity += pnl
                daily_realized[day_key] = daily_realized.get(day_key, 0.0) + total_pnl
                weekly_realized[week_key] = weekly_realized.get(week_key, 0.0) + total_pnl
                consecutive_losses = consecutive_losses + 1 if total_pnl < 0 else 0
                open_pos = None
                continue

            if not bool(open_pos["tp1_hit"]):
                tp1_hit = (high >= tp1_price) if side == "LONG" else (low <= tp1_price)
                if tp1_hit:
                    qty = min(float(open_pos["entry_qty"]) * cfg.tp1_pct, float(open_pos["remaining_qty"]))
                    pnl = _close_piece(side, entry_price, tp1_price, qty)
                    r_piece = pnl / max(1e-12, risk_per_unit * qty)
                    open_pos["remaining_qty"] = float(open_pos["remaining_qty"]) - qty
                    open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                    open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                    open_pos["tp1_hit"] = True
                    equity += pnl

            if not bool(open_pos["tp2_hit"]):
                tp2_hit = (high >= tp2_price) if side == "LONG" else (low <= tp2_price)
                if tp2_hit:
                    qty = min(float(open_pos["entry_qty"]) * cfg.tp2_pct, float(open_pos["remaining_qty"]))
                    pnl = _close_piece(side, entry_price, tp2_price, qty)
                    r_piece = pnl / max(1e-12, risk_per_unit * qty)
                    open_pos["remaining_qty"] = float(open_pos["remaining_qty"]) - qty
                    open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                    open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                    open_pos["tp2_hit"] = True
                    equity += pnl

            if bool(open_pos["tp1_hit"]) and bool(open_pos["tp2_hit"]) and float(open_pos["remaining_qty"]) > 0:
                trailed = strategy.trailing_stop(side, hist15, float(open_pos["stop_loss"]))
                if (side == "LONG" and trailed > float(open_pos["stop_loss"])) or (
                    side == "SHORT" and trailed < float(open_pos["stop_loss"])
                ):
                    open_pos["stop_loss"] = trailed
                stop_loss = float(open_pos["stop_loss"])
                trail_hit = (low <= stop_loss) if side == "LONG" else (high >= stop_loss)
                if trail_hit:
                    qty = float(open_pos["remaining_qty"])
                    pnl = _close_piece(side, entry_price, stop_loss, qty)
                    r_piece = pnl / max(1e-12, risk_per_unit * qty)
                    open_pos["remaining_qty"] = 0.0
                    open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                    open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                    total_pnl = float(open_pos["realized_pnl"])
                    equity += pnl
                    daily_realized[day_key] = daily_realized.get(day_key, 0.0) + total_pnl
                    weekly_realized[week_key] = weekly_realized.get(week_key, 0.0) + total_pnl
                    consecutive_losses = consecutive_losses + 1 if total_pnl < 0 else 0
                    open_pos = None
                    continue

            if open_pos is not None and strategy.is_opposite_signal(side, hist15, hist1h):
                qty = float(open_pos["remaining_qty"])
                if qty > 0:
                    pnl = _close_piece(side, entry_price, close, qty)
                    r_piece = pnl / max(1e-12, risk_per_unit * qty)
                    open_pos["remaining_qty"] = 0.0
                    open_pos["realized_pnl"] = float(open_pos["realized_pnl"]) + pnl
                    open_pos["realized_r"] = float(open_pos["realized_r"]) + r_piece
                    equity += pnl
                total_pnl = float(open_pos["realized_pnl"])
                daily_realized[day_key] = daily_realized.get(day_key, 0.0) + total_pnl
                weekly_realized[week_key] = weekly_realized.get(week_key, 0.0) + total_pnl
                consecutive_losses = consecutive_losses + 1 if total_pnl < 0 else 0
                open_pos = None
                continue

            continue

        evaluated_entry_candles += 1
        daily_pnl = daily_realized.get(day_key, 0.0)
        weekly_pnl = weekly_realized.get(week_key, 0.0)

        blocked_reason = ""
        if daily_pnl <= -abs(equity * cfg.daily_loss_limit_pct):
            blocked_reason = "daily_loss_limit_reached"
        elif weekly_pnl <= -abs(equity * cfg.weekly_loss_limit_pct):
            blocked_reason = "weekly_loss_limit_reached"
        elif consecutive_losses >= cfg.max_consecutive_losses:
            blocked_reason = "max_consecutive_losses_reached"
        elif trades_today.get(day_key, 0) >= cfg.max_trades_per_day:
            blocked_reason = "max_trades_per_day_reached"

        if blocked_reason:
            risk_gate["blocked_checks"] += 1
            reasons = risk_gate["blocked_reasons"]
            reasons[blocked_reason] = int(reasons.get(blocked_reason, 0)) + 1
            pending_signal = "NONE"
            blocked_sig = strategy.entry_signal(hist15, hist1h)
            if blocked_sig.side != "NONE":
                _add(counts["risk_blocked_signals"], blocked_sig.side)
            continue

        risk_gate["allowed_checks"] += 1
        signal_out = strategy.entry_signal(hist15, hist1h)
        if signal_out.side == "NONE":
            pending_signal = "NONE"
            continue

        _add(counts["raw_strategy_signals"], signal_out.side)

        if pending_signal != signal_out.side:
            pending_signal = signal_out.side
            continue

        pending_signal = "NONE"
        _add(counts["confirmed_signals"], signal_out.side)

        allow_ai = True
        if ai_filter.enabled:
            features = build_runtime_feature_vector(hist15, hist1h, cfg, signal_out.side)
            if features is None:
                allow_ai = False
            else:
                allow_ai = ai_filter.decide(features).allow

        if not allow_ai:
            _add(counts["ai_rejected_signals"], signal_out.side)
            continue

        _add(counts["ai_accepted_signals"], signal_out.side)
        _add(counts["risk_allowed_signals"], signal_out.side)

        atr_value = signal_out.atr
        if atr_value <= 0:
            continue

        if signal_out.side == "LONG":
            stop = close - cfg.atr_stop_mult * atr_value
            tp1 = close + abs(close - stop) * cfg.tp1_r
            tp2 = close + abs(close - stop) * cfg.tp2_r
        else:
            stop = close + cfg.atr_stop_mult * atr_value
            tp1 = close - abs(close - stop) * cfg.tp1_r
            tp2 = close - abs(close - stop) * cfg.tp2_r

        risk_per_unit = abs(close - stop)
        if risk_per_unit <= 0:
            continue

        risk_amount = equity * cfg.risk_per_trade_pct
        qty_by_risk = risk_amount / risk_per_unit
        qty_by_lev = (equity * cfg.leverage) / close if close > 0 else 0.0
        qty = min(qty_by_risk, qty_by_lev)
        if qty <= 0:
            continue

        open_pos = {
            "side": signal_out.side,
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
        _add(counts["trades_opened"], signal_out.side)

    def _with_freq(bucket: dict[str, int]) -> dict[str, float | int]:
        out = dict(bucket)
        out["per_100_candles"] = round((bucket["total"] / evaluated_entry_candles * 100.0), 4) if evaluated_entry_candles else 0.0
        return out

    report = {
        "window_15m": window,
        "evaluated_entry_candles": evaluated_entry_candles,
        "ai_filter_enabled": bool(ai_filter.enabled),
        "ai_threshold": float(ai_filter.threshold) if ai_filter.enabled else None,
        "gate_counts": {
            "raw_strategy_signals": _with_freq(counts["raw_strategy_signals"]),
            "confirmed_signals": _with_freq(counts["confirmed_signals"]),
            "ai_accepted_signals": _with_freq(counts["ai_accepted_signals"]),
            "ai_rejected_signals": _with_freq(counts["ai_rejected_signals"]),
            "risk_allowed_signals": _with_freq(counts["risk_allowed_signals"]),
            "risk_blocked_signals": _with_freq(counts["risk_blocked_signals"]),
            "trades_opened": _with_freq(counts["trades_opened"]),
        },
        "risk_gate": risk_gate,
    }
    if emit_json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    return report


def main() -> None:
    args = parse_args()
    cfg = BotConfig.from_env()

    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    command = args.command or "run-bot"
    try:
        if command == "run-bot":
            run_bot(
                cfg,
                once=bool(getattr(args, "once", False)),
                paper=bool(getattr(args, "paper", False)),
            )
            return

        if command == "backtest":
            run_backtest(
                cfg,
                limit=int(args.limit),
                candles15_csv=args.candles15_csv,
                candles1h_csv=args.candles1h_csv,
                output=args.output,
                use_ai_filter=bool(args.use_ai_filter),
                compare_ai_filter=bool(args.compare_ai_filter),
            )
            return

        if command == "generate-dataset":
            run_generate_dataset(
                cfg,
                limit=int(args.limit),
                output=args.output,
                candles15_csv=args.candles15_csv,
                candles1h_csv=args.candles1h_csv,
            )
            return

        if command == "train-ai":
            run_train_ai(dataset=args.dataset, model_output=args.model_output)
            return

        if command == "train-model":
            run_train_model(dataset=args.dataset, output=args.output)
            return

        if command == "paper-report":
            run_paper_report(cfg)
            return

        if command == "diagnose-runtime":
            raw_thresholds = str(getattr(args, "ai_thresholds", "") or "").strip()
            if raw_thresholds:
                thresholds = [float(x.strip()) for x in raw_thresholds.split(",") if x.strip()]
                reports = []
                for threshold in thresholds:
                    cfg_threshold = replace(cfg, ai_filter_enabled=True, ai_score_threshold=threshold)
                    rep = run_diagnose_runtime(
                        cfg_threshold,
                        window=int(args.window),
                        candles15_csv=args.candles15_csv,
                        candles1h_csv=args.candles1h_csv,
                        emit_json=False,
                    )
                    reports.append(
                        {
                            "threshold": threshold,
                            "evaluated_entry_candles": rep["evaluated_entry_candles"],
                            "raw_strategy_signals": rep["gate_counts"]["raw_strategy_signals"],
                            "confirmed_signals": rep["gate_counts"]["confirmed_signals"],
                            "ai_accepted_signals": rep["gate_counts"]["ai_accepted_signals"],
                            "ai_rejected_signals": rep["gate_counts"]["ai_rejected_signals"],
                            "trades_opened": rep["gate_counts"]["trades_opened"],
                        }
                    )
                print(
                    json.dumps(
                        {
                            "window_15m": int(args.window),
                            "ai_filter_forced_enabled": True,
                            "threshold_reports": reports,
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )
                return

            run_diagnose_runtime(
                cfg,
                window=int(args.window),
                candles15_csv=args.candles15_csv,
                candles1h_csv=args.candles1h_csv,
            )
            return

        raise ValueError(f"Unsupported command: {command}")
    except Exception as exc:  # noqa: BLE001
        logging.error("Command failed: %s", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
