from __future__ import annotations

import logging
import signal
import threading
from datetime import datetime
from types import FrameType

from .ai_filter import AISignalFilter, build_runtime_feature_vector
from .config import BotConfig
from .db import Database
from .exchange import BybitClient, InstrumentRules
from .models import FuturesPosition, Side
from .notifier import TelegramNotifier
from .risk import RiskManager
from .strategy import BreakoutMomentumStrategy


class TradingBot:
    def __init__(self, cfg: BotConfig, paper_mode: bool = False) -> None:
        self.cfg = cfg
        self.paper_mode = paper_mode
        self.db = Database(cfg.db_path)
        self.exchange = BybitClient(cfg.api_key, cfg.api_secret, cfg.bybit_testnet)
        self.strategy = BreakoutMomentumStrategy(cfg)
        self.risk = RiskManager(cfg, self.db)
        self.notifier = TelegramNotifier(cfg.telegram_bot_token, cfg.telegram_chat_id)
        self.ai_filter = AISignalFilter(
            enabled=cfg.ai_filter_enabled,
            model_path=cfg.ai_model_path,
            threshold=cfg.ai_score_threshold,
        )
        if self.paper_mode and self.ai_filter.enabled:
            self.ai_filter.validate_runtime_schema()

        self._logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._trading_halted = False
        self._halt_reason = ""
        self._pending_signal_side = "NONE"

    def run(self, once: bool = False) -> None:
        self._install_signal_handlers()
        if self.paper_mode:
            mode = "PAPER"
        else:
            mode = "DRY-RUN" if self.cfg.dry_run else "LIVE"
        network = "TESTNET" if self.cfg.bybit_testnet else "MAINNET"

        self._log("INFO", f"Bot started | {mode} | {network} | {self.cfg.symbol}")
        tag = "[PAPER]" if self.paper_mode else "[BOT]"
        self.notifier.send(f"{tag} Started {mode} on {network} for {self.cfg.symbol}")

        if not self.cfg.dry_run and not self.paper_mode:
            try:
                self.exchange.set_leverage(self.cfg.symbol, self.cfg.leverage)
            except Exception as exc:  # noqa: BLE001
                self._log("ERROR", f"Failed to set leverage: {exc}")
                self.notifier.send(f"[BOT] Error setting leverage: {exc}")

        self._reconcile_state()

        try:
            while not self._stop_event.is_set():
                try:
                    self._tick()
                except Exception as exc:  # noqa: BLE001
                    self._log("ERROR", f"Tick failed: {exc}")
                    self.notifier.send(f"[BOT] Error: {exc}")

                if once:
                    break

                self._stop_event.wait(self.cfg.poll_interval_sec)
        finally:
            self.db.close()

    def stop(self) -> None:
        self._stop_event.set()

    def _install_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except ValueError:
            pass

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        self._log("INFO", f"Signal received: {signum}")
        self.stop()

    def _tick(self) -> None:
        candles_15m_raw = self._fetch_candles(self.cfg.signal_interval, self.cfg.signal_candle_limit)
        candles_1h_raw = self._fetch_candles(self.cfg.trend_interval, self.cfg.trend_candle_limit)
        if candles_15m_raw.empty or candles_1h_raw.empty:
            self._log("WARNING", "No candle data available")
            return

        candles_15m, dropped_open_15m = self._exclude_forming_last_bar(candles_15m_raw, self.cfg.signal_interval)
        candles_1h, dropped_open_1h = self._exclude_forming_last_bar(candles_1h_raw, self.cfg.trend_interval)
        if candles_15m.empty or candles_1h.empty:
            self._log("WARNING", "No closed candle data available after trimming forming bars")
            return

        self._log(
            "INFO",
            (
                f"Evaluating closed candles: "
                f"15m_ts={int(candles_15m.iloc[-1]['ts'])} "
                f"1h_ts={int(candles_1h.iloc[-1]['ts'])} "
                f"dropped_open_15m={dropped_open_15m} "
                f"dropped_open_1h={dropped_open_1h}"
            ),
        )

        last_price = self._safe_last_price(float(candles_15m_raw.iloc[-1]["close"]))
        rules = self.exchange.get_instrument_rules(self.cfg.symbol)

        open_trade = self.db.get_open_trade(self.cfg.symbol)
        if open_trade is not None:
            self._manage_open_trade(open_trade, candles_15m, candles_1h, last_price, rules)
            return

        equity = self._equity_estimate()
        gate = self.risk.check_new_trade_allowed(equity)
        if not gate.allowed:
            self._set_halt(gate.reason)
            return

        if self._trading_halted:
            self._trading_halted = False
            self._halt_reason = ""
            self._log("INFO", "Trading resumed after halt")
            self.notifier.send("[BOT] Trading resumed")

        signal_out = self.strategy.entry_signal(candles_15m, candles_1h)
        if signal_out.side == "NONE":
            self._pending_signal_side = "NONE"
            self._log("INFO", f"No entry signal: {signal_out.reason}")
            return

        if self._pending_signal_side != signal_out.side:
            self._pending_signal_side = signal_out.side
            self._log("INFO", f"Signal detected, waiting next cycle confirmation: {signal_out.side}")
            return

        self._pending_signal_side = "NONE"

        stop_loss, tp1_price, tp2_price, _ = self._build_levels(
            side=signal_out.side,
            entry_price=last_price,
            atr_value=signal_out.atr,
            tick_size=rules.tick_size,
        )

        score: float | None = None
        allow_signal = True
        reject_reason = ""
        if self.ai_filter.enabled:
            features = build_runtime_feature_vector(
                candles_15m=candles_15m,
                candles_1h=candles_1h,
                cfg=self.cfg,
                side=signal_out.side,
            )
            if features is None:
                self._log("WARNING", "AI filter enabled but feature vector could not be built")
                allow_signal = False
                reject_reason = "ai_features_unavailable"
            else:
                decision = self.ai_filter.decide(features)
                score = decision.score
                allow_signal = decision.allow
                if not allow_signal:
                    reject_reason = "ai_rejected"

        candidate_ts = int(candles_15m.iloc[-1]["ts"])
        candidate_data = {
            "event": "paper_candidate",
            "paper_mode": self.paper_mode,
            "timestamp": candidate_ts,
            "symbol": self.cfg.symbol,
            "side": signal_out.side,
            "model_probability": score,
            "allowed": allow_signal,
            "planned_entry": last_price,
            "planned_stop": stop_loss,
            "planned_target": tp2_price,
            "planned_target_tp1": tp1_price,
            "planned_target_tp2": tp2_price,
            "reason": reject_reason or signal_out.reason,
        }
        self._log(
            "INFO",
            (
                f"Candidate {signal_out.side} ts={candidate_ts} prob="
                f"{'NA' if score is None else f'{score:.4f}'} allowed={allow_signal} "
                f"entry={last_price:.2f} stop={stop_loss:.2f} tp1={tp1_price:.2f} tp2={tp2_price:.2f}"
            ),
            data=candidate_data,
        )
        if self.paper_mode:
            self.notifier.send(
                (
                    f"[PAPER] Candidate {signal_out.side} {self.cfg.symbol}\n"
                    f"ts={candidate_ts} prob={'NA' if score is None else f'{score:.4f}'} "
                    f"allowed={allow_signal}\n"
                    f"entry={last_price:.2f} stop={stop_loss:.2f} "
                    f"target={tp2_price:.2f}"
                )
            )

        if not allow_signal:
            self._log(
                "INFO",
                f"Paper decision rejected side={signal_out.side} prob={'NA' if score is None else f'{score:.4f}'}",
                data={
                    "event": "paper_decision",
                    "paper_mode": self.paper_mode,
                    "decision": "rejected",
                    "timestamp": candidate_ts,
                    "symbol": self.cfg.symbol,
                    "side": signal_out.side,
                    "model_probability": score,
                },
            )
            if score is not None:
                self._log("INFO", f"Signal rejected by AI filter side={signal_out.side} score={score:.4f}")
            if self.paper_mode:
                self.notifier.send(
                    (
                        f"[PAPER] Trade rejected by model {signal_out.side}\n"
                        f"prob={'NA' if score is None else f'{score:.4f}'} reason={reject_reason or 'rejected'}"
                    )
                )
            return

        self._log(
            "INFO",
            f"Paper decision accepted side={signal_out.side} prob={'NA' if score is None else f'{score:.4f}'}",
            data={
                "event": "paper_decision",
                "paper_mode": self.paper_mode,
                "decision": "accepted",
                "timestamp": candidate_ts,
                "symbol": self.cfg.symbol,
                "side": signal_out.side,
                "model_probability": score,
            },
        )

        if self.paper_mode:
            self.notifier.send(
                (
                    f"[PAPER] Trade accepted {signal_out.side} {self.cfg.symbol}\n"
                    f"prob={'NA' if score is None else f'{score:.4f}'} "
                    f"entry={last_price:.2f} stop={stop_loss:.2f} target={tp2_price:.2f}"
                )
            )

        self._open_new_trade(signal_out.side, signal_out.atr, last_price, rules, ai_score=score)

    def _open_new_trade(
        self,
        side: str,
        atr_value: float,
        entry_price: float,
        rules: InstrumentRules,
        ai_score: float | None = None,
    ) -> None:
        if atr_value <= 0:
            self._log("WARNING", "ATR is non-positive, cannot open trade")
            return

        stop_loss, tp1_price, tp2_price, risk_per_unit = self._build_levels(
            side=side,
            entry_price=entry_price,
            atr_value=atr_value,
            tick_size=rules.tick_size,
        )

        equity = self._equity_estimate()
        qty = self.risk.calculate_position_size(
            equity_usdt=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            min_qty=rules.min_qty,
            qty_step=rules.qty_step,
            min_notional=rules.min_notional,
        )
        qty = self.exchange.round_qty_down(qty, rules.qty_step)

        if qty <= 0:
            self._log("WARNING", "Position size became zero after precision filters")
            return

        order_side = "Buy" if side == "LONG" else "Sell"
        external_id = "paper" if self.paper_mode else "dryrun"
        if not self.cfg.dry_run and not self.paper_mode:
            external_id = self.exchange.place_market_order(
                symbol=self.cfg.symbol,
                side=order_side,
                qty=qty,
                reduce_only=False,
            )

        self.db.log_order(
            symbol=self.cfg.symbol,
            side=order_side.upper(),
            qty=qty,
            price=entry_price,
            reduce_only=False,
            status="FILLED",
            dry_run=self.cfg.dry_run,
            external_id=external_id,
            notes="ENTRY",
        )
        trade_id = self.db.create_trade(
            symbol=self.cfg.symbol,
            side=side,
            entry_qty=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            risk_per_unit=risk_per_unit,
        )

        self._log(
            "INFO",
            (
                f"Opened {side} #{trade_id} qty={qty:.6f} entry={entry_price:.2f} "
                f"sl={stop_loss:.2f} tp1={tp1_price:.2f} tp2={tp2_price:.2f}"
            ),
            data={
                "event": "paper_trade_open" if self.paper_mode else "trade_open",
                "trade_id": trade_id,
                "symbol": self.cfg.symbol,
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "tp1_price": tp1_price,
                "tp2_price": tp2_price,
                "qty": qty,
                "paper_mode": self.paper_mode,
                "ai_probability": ai_score,
                "paper_equity": self._equity_estimate(),
            },
        )
        self.notifier.send(
            (
                f"{'[PAPER]' if self.paper_mode else '[BOT]'} Opened {side} #{trade_id}\n"
                f"qty={qty:.6f} entry={entry_price:.2f}\n"
                f"SL={stop_loss:.2f} TP1={tp1_price:.2f} TP2={tp2_price:.2f}"
            )
        )

    def _manage_open_trade(
        self,
        pos: FuturesPosition,
        candles_15m,
        candles_1h,
        last_price: float,
        rules: InstrumentRules,
    ) -> None:
        if self._is_stop_hit(pos, last_price):
            self._exit_piece(pos, pos.remaining_qty, last_price, "STOP_LOSS", rules, tp_level=0)
            return

        current = pos

        if (not current.tp1_hit) and self._is_tp1_hit(current, last_price):
            qty = self.exchange.round_qty_down(current.entry_qty * self.cfg.tp1_pct, rules.qty_step)
            qty = min(qty, current.remaining_qty)
            if qty > 0:
                self._exit_piece(current, qty, last_price, "TP1", rules, tp_level=1)
            current = self.db.get_open_trade(self.cfg.symbol)
            if current is None:
                return

        if (not current.tp2_hit) and self._is_tp2_hit(current, last_price):
            qty = self.exchange.round_qty_down(current.entry_qty * self.cfg.tp2_pct, rules.qty_step)
            qty = min(qty, current.remaining_qty)
            if qty > 0:
                self._exit_piece(current, qty, last_price, "TP2", rules, tp_level=2)
            current = self.db.get_open_trade(self.cfg.symbol)
            if current is None:
                return

        if current.tp1_hit and current.tp2_hit:
            trailed = self.strategy.trailing_stop(current.side, candles_15m, current.stop_loss)
            mode = "down" if current.side == "LONG" else "up"
            trailed = self.exchange.round_price(trailed, rules.tick_size, mode=mode)

            better = (current.side == "LONG" and trailed > current.stop_loss) or (
                current.side == "SHORT" and trailed < current.stop_loss
            )
            if better:
                self.db.update_stop_loss(current.trade_id, trailed)
                current = self.db.get_open_trade(self.cfg.symbol)
                if current is None:
                    return
                self._log("INFO", f"Updated trailing stop to {trailed:.2f} for trade #{current.trade_id}")

            if self._is_stop_hit(current, last_price):
                self._exit_piece(current, current.remaining_qty, last_price, "TRAIL_STOP", rules, tp_level=0)
                return

        if self.strategy.is_opposite_signal(current.side, candles_15m, candles_1h):
            self._exit_piece(current, current.remaining_qty, last_price, "OPPOSITE_SIGNAL", rules, tp_level=0)
            return

        self._log(
            "INFO",
            (
                f"Holding {current.side} #{current.trade_id} qty={current.remaining_qty:.6f} "
                f"price={last_price:.2f} stop={current.stop_loss:.2f}"
            ),
        )

    def _exit_piece(
        self,
        pos: FuturesPosition,
        qty: float,
        price: float,
        reason: str,
        rules: InstrumentRules,
        tp_level: int,
    ) -> None:
        qty = self.exchange.round_qty_down(qty, rules.qty_step)
        if qty <= 0:
            return

        order_side = "Sell" if pos.side == "LONG" else "Buy"
        external_id = "paper" if self.paper_mode else "dryrun"
        if not self.cfg.dry_run and not self.paper_mode:
            external_id = self.exchange.place_market_order(
                symbol=self.cfg.symbol,
                side=order_side,
                qty=qty,
                reduce_only=True,
            )

        self.db.log_order(
            symbol=self.cfg.symbol,
            side=order_side.upper(),
            qty=qty,
            price=price,
            reduce_only=True,
            status="FILLED",
            dry_run=self.cfg.dry_run,
            external_id=external_id,
            notes=reason,
        )

        event_type = "PARTIAL_EXIT" if tp_level in {1, 2} else "FINAL_EXIT"
        result = self.db.apply_exit(
            trade_id=pos.trade_id,
            exit_qty=qty,
            exit_price=price,
            event_type=event_type,
            notes=reason,
            fee=0.0,
        )

        if tp_level in {1, 2}:
            self.db.mark_tp_hit(pos.trade_id, tp_level)
            self.notifier.send(
                (
                    f"[BOT] Partial TP{tp_level} hit for trade #{pos.trade_id}\n"
                    f"qty={qty:.6f} price={price:.2f} pnl={float(result['pnl']):.2f}"
                )
            )

        if bool(result["closed"]):
            snap = self.db.get_trade_snapshot(pos.trade_id) or {}
            total_pnl = float(snap.get("realized_pnl", result["pnl"]))
            total_r = float(snap.get("realized_r", result["r_multiple"]))
            cumulative_equity = self._equity_estimate()
            opened_ts = str(snap.get("opened_ts", ""))
            closed_ts = str(snap.get("closed_ts", ""))
            self._log(
                "INFO",
                (
                    f"Closed trade #{pos.trade_id} reason={reason} "
                    f"pnl_total={total_pnl:.2f} r_total={total_r:.4f} "
                    f"paper_equity={cumulative_equity:.2f}"
                ),
                data={
                    "event": "paper_trade_closed" if self.paper_mode else "trade_closed",
                    "trade_id": pos.trade_id,
                    "reason": reason,
                    "realized_pnl": total_pnl,
                    "realized_r": total_r,
                    "opened_ts": opened_ts,
                    "closed_ts": closed_ts,
                    "paper_equity": cumulative_equity,
                    "paper_mode": self.paper_mode,
                },
            )
            if self.paper_mode:
                self.notifier.send(
                    (
                        f"[PAPER] Trade closed #{pos.trade_id}\n"
                        f"entry_time={opened_ts}\n"
                        f"exit_time={closed_ts}\n"
                        f"result_r={total_r:.4f} total_pnl={total_pnl:.2f}\n"
                        f"paper_equity={cumulative_equity:.2f}"
                    )
                )
            else:
                self.notifier.send(
                    (
                        f"[BOT] Final close #{pos.trade_id}\n"
                        f"reason={reason} total_pnl={total_pnl:.2f}"
                    )
                )

    def _is_stop_hit(self, pos: FuturesPosition, price: float) -> bool:
        if pos.side == "LONG":
            return price <= pos.stop_loss
        return price >= pos.stop_loss

    def _is_tp1_hit(self, pos: FuturesPosition, price: float) -> bool:
        if pos.side == "LONG":
            return price >= pos.tp1_price
        return price <= pos.tp1_price

    def _is_tp2_hit(self, pos: FuturesPosition, price: float) -> bool:
        if pos.side == "LONG":
            return price >= pos.tp2_price
        return price <= pos.tp2_price

    def _build_levels(self, side: str, entry_price: float, atr_value: float, tick_size: float) -> tuple[float, float, float, float]:
        risk_per_unit = atr_value * self.cfg.atr_stop_mult

        if side == "LONG":
            stop = entry_price - risk_per_unit
            tp1 = entry_price + risk_per_unit * self.cfg.tp1_r
            tp2 = entry_price + risk_per_unit * self.cfg.tp2_r
            stop = self.exchange.round_price(stop, tick_size, mode="down")
            tp1 = self.exchange.round_price(tp1, tick_size, mode="up")
            tp2 = self.exchange.round_price(tp2, tick_size, mode="up")
        else:
            stop = entry_price + risk_per_unit
            tp1 = entry_price - risk_per_unit * self.cfg.tp1_r
            tp2 = entry_price - risk_per_unit * self.cfg.tp2_r
            stop = self.exchange.round_price(stop, tick_size, mode="up")
            tp1 = self.exchange.round_price(tp1, tick_size, mode="down")
            tp2 = self.exchange.round_price(tp2, tick_size, mode="down")

        return stop, tp1, tp2, abs(entry_price - stop)

    def _reconcile_state(self) -> None:
        local = self.db.get_open_trade(self.cfg.symbol)

        if self.cfg.dry_run:
            if local is not None:
                self._log("INFO", f"Recovered local open trade #{local.trade_id} (dry-run mode)")
            return

        try:
            exch = self.exchange.get_open_position(self.cfg.symbol)
        except Exception as exc:  # noqa: BLE001
            self._log("WARNING", f"Exchange reconciliation skipped: {exc}")
            return

        price = self._safe_last_price(None)

        if exch is None and local is None:
            return

        if exch is None and local is not None:
            self.db.force_close_open_trade(self.cfg.symbol, price or local.entry_price, "reconcile_exchange_flat")
            self._log("WARNING", "Local open trade closed because exchange has no position")
            self.notifier.send("[BOT] Reconciliation: local position closed because exchange is flat")
            return

        if exch is not None and local is None:
            atr_value = self._latest_atr_fallback()
            rules = self.exchange.get_instrument_rules(self.cfg.symbol)
            stop, tp1, tp2, risk_unit = self._build_levels(exch.side, exch.entry_price, atr_value, rules.tick_size)
            trade_id = self.db.create_recovered_trade(
                symbol=self.cfg.symbol,
                side=exch.side,
                qty=exch.qty,
                entry_price=exch.entry_price,
                stop_loss=stop,
                tp1_price=tp1,
                tp2_price=tp2,
                risk_per_unit=risk_unit,
            )
            self._log("WARNING", f"Recovered exchange position as local trade #{trade_id}")
            self.notifier.send(f"[BOT] Recovered exchange position into local trade #{trade_id}")
            return

        assert exch is not None
        assert local is not None
        qty_diff = abs(local.remaining_qty - exch.qty)
        if local.side != exch.side or qty_diff > max(1e-8, exch.qty * 0.01):
            self.db.force_close_open_trade(self.cfg.symbol, price or local.entry_price, "reconcile_mismatch")
            atr_value = self._latest_atr_fallback()
            rules = self.exchange.get_instrument_rules(self.cfg.symbol)
            stop, tp1, tp2, risk_unit = self._build_levels(exch.side, exch.entry_price, atr_value, rules.tick_size)
            trade_id = self.db.create_recovered_trade(
                symbol=self.cfg.symbol,
                side=exch.side,
                qty=exch.qty,
                entry_price=exch.entry_price,
                stop_loss=stop,
                tp1_price=tp1,
                tp2_price=tp2,
                risk_per_unit=risk_unit,
            )
            self._log("WARNING", f"Reconciled mismatch and recovered trade #{trade_id}")
            self.notifier.send(f"[BOT] Reconciled mismatch and recovered trade #{trade_id}")

    def _latest_atr_fallback(self) -> float:
        candles = self._fetch_candles(self.cfg.signal_interval, self.cfg.signal_candle_limit)
        trend = self._fetch_candles(self.cfg.trend_interval, self.cfg.trend_candle_limit)
        signal_out = self.strategy.entry_signal(candles, trend)
        if signal_out.atr > 0:
            return signal_out.atr
        if not candles.empty:
            return float(candles.iloc[-1]["close"]) * 0.005
        return 100.0

    def _fetch_candles(self, interval: str, limit: int):
        try:
            df = self.exchange.get_klines(symbol=self.cfg.symbol, interval=interval, limit=limit)
            if not df.empty:
                self.db.upsert_candles(self.cfg.symbol, interval, df)
                return df
        except Exception as exc:  # noqa: BLE001
            self._log("WARNING", f"REST candle fetch failed ({interval}), trying cache: {exc}")

        cached = self.db.get_cached_candles(self.cfg.symbol, interval, limit)
        return cached

    def _exclude_forming_last_bar(self, candles, interval: str):
        if candles.empty:
            return candles, False

        try:
            interval_min = int(interval)
        except ValueError:
            return candles, False

        now_ms = int(datetime.utcnow().timestamp() * 1000)
        last_ts = int(candles.iloc[-1]["ts"])
        interval_ms = interval_min * 60_000
        is_forming = (last_ts + interval_ms) > now_ms

        if is_forming and len(candles) > 1:
            return candles.iloc[:-1].reset_index(drop=True), True
        return candles, False

    def _safe_last_price(self, fallback_close: float | None) -> float:
        try:
            return self.exchange.get_last_price(self.cfg.symbol)
        except Exception as exc:  # noqa: BLE001
            self._log("WARNING", f"Ticker fetch failed, fallback to candle close: {exc}")
            if fallback_close is not None:
                return fallback_close
            return 0.0

    def _equity_estimate(self) -> float:
        if not self.cfg.dry_run:
            try:
                return self.exchange.get_usdt_balance()
            except Exception as exc:  # noqa: BLE001
                self._log("WARNING", f"Balance fetch failed, using local estimate: {exc}")

        realized = self.db.get_total_realized_pnl()
        return max(0.0, self.cfg.starting_capital_usdt + realized)

    def _set_halt(self, reason: str) -> None:
        if self._trading_halted and self._halt_reason == reason:
            return

        self._trading_halted = True
        self._halt_reason = reason
        self._log("WARNING", f"Trading halted: {reason}")
        self.notifier.send(f"[BOT] Trading halted: {reason}")

    def _log(self, level: str, message: str, data: dict | None = None) -> None:
        self.db.log_event(level, message, data)
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(message)
