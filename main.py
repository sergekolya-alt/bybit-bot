from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from bot.ai_filter import AISignalFilter, build_signal_dataset, train_ai_model, train_rf_model
from bot.backtest import StrategyBacktester, fetch_historical_klines, load_candles_csv
from bot.config import BotConfig
from bot.db import Database
from bot.engine import TradingBot
from bot.exchange import BybitClient


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

        raise ValueError(f"Unsupported command: {command}")
    except Exception as exc:  # noqa: BLE001
        logging.error("Command failed: %s", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
