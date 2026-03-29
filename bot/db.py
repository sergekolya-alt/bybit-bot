from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .models import FuturesPosition, Side


class Database:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                reduce_only INTEGER NOT NULL,
                status TEXT NOT NULL,
                dry_run INTEGER NOT NULL,
                external_id TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_qty REAL NOT NULL,
                remaining_qty REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                tp1_price REAL NOT NULL,
                tp2_price REAL NOT NULL,
                risk_per_unit REAL NOT NULL,
                tp1_hit INTEGER NOT NULL DEFAULT 0,
                tp2_hit INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'OPEN',
                opened_ts TEXT NOT NULL,
                closed_ts TEXT,
                close_reason TEXT,
                realized_pnl REAL NOT NULL DEFAULT 0,
                realized_r REAL NOT NULL DEFAULT 0,
                fees REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL NOT NULL DEFAULT 0,
                r_multiple REAL NOT NULL DEFAULT 0,
                fee REAL NOT NULL DEFAULT 0,
                notes TEXT,
                FOREIGN KEY(trade_id) REFERENCES trades(id)
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT
            );

            CREATE TABLE IF NOT EXISTS candle_cache (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                ts INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, interval, ts)
            );

            CREATE INDEX IF NOT EXISTS idx_trade_status_symbol
            ON trades (symbol, status, id DESC);

            CREATE INDEX IF NOT EXISTS idx_trade_closed_symbol
            ON trades (symbol, closed_ts DESC);

            CREATE INDEX IF NOT EXISTS idx_candle_cache_lookup
            ON candle_cache (symbol, interval, ts DESC);
            """
        )
        self.conn.commit()

    def log_event(self, level: str, message: str, data: dict[str, Any] | None = None) -> None:
        ts = datetime.utcnow().isoformat(timespec="seconds")
        payload = json.dumps(data or {}, ensure_ascii=True)
        self.conn.execute(
            "INSERT INTO events (ts, level, message, data) VALUES (?, ?, ?, ?)",
            (ts, level.upper(), message, payload),
        )
        self.conn.commit()

    def log_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reduce_only: bool,
        status: str,
        dry_run: bool,
        external_id: str = "",
        notes: str = "",
    ) -> int:
        ts = datetime.utcnow().isoformat(timespec="seconds")
        cur = self.conn.execute(
            """
            INSERT INTO orders (ts, symbol, side, qty, price, reduce_only, status, dry_run, external_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, symbol, side, qty, price, int(reduce_only), status, int(dry_run), external_id, notes),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def create_trade(
        self,
        symbol: str,
        side: Side,
        entry_qty: float,
        entry_price: float,
        stop_loss: float,
        tp1_price: float,
        tp2_price: float,
        risk_per_unit: float,
    ) -> int:
        opened_ts = datetime.utcnow().isoformat(timespec="seconds")
        cur = self.conn.execute(
            """
            INSERT INTO trades (
                symbol, side, entry_qty, remaining_qty, entry_price,
                stop_loss, tp1_price, tp2_price, risk_per_unit, opened_ts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                side,
                entry_qty,
                entry_qty,
                entry_price,
                stop_loss,
                tp1_price,
                tp2_price,
                risk_per_unit,
                opened_ts,
            ),
        )
        trade_id = int(cur.lastrowid)
        self._insert_trade_event(
            trade_id=trade_id,
            event_type="ENTRY",
            qty=entry_qty,
            price=entry_price,
            pnl=0.0,
            r_multiple=0.0,
            fee=0.0,
            notes="entry",
        )
        self.conn.commit()
        return trade_id

    def create_recovered_trade(
        self,
        symbol: str,
        side: Side,
        qty: float,
        entry_price: float,
        stop_loss: float,
        tp1_price: float,
        tp2_price: float,
        risk_per_unit: float,
    ) -> int:
        trade_id = self.create_trade(
            symbol=symbol,
            side=side,
            entry_qty=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            risk_per_unit=risk_per_unit,
        )
        self._insert_trade_event(
            trade_id=trade_id,
            event_type="RECONCILE_RECOVER",
            qty=qty,
            price=entry_price,
            pnl=0.0,
            r_multiple=0.0,
            fee=0.0,
            notes="Recovered from exchange position",
        )
        self.conn.commit()
        return trade_id

    def get_open_trade(self, symbol: str) -> FuturesPosition | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM trades
            WHERE symbol = ? AND status = 'OPEN'
            ORDER BY id DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if row is None:
            return None

        return FuturesPosition(
            trade_id=int(row["id"]),
            symbol=row["symbol"],
            side=row["side"],
            entry_qty=float(row["entry_qty"]),
            remaining_qty=float(row["remaining_qty"]),
            entry_price=float(row["entry_price"]),
            stop_loss=float(row["stop_loss"]),
            tp1_price=float(row["tp1_price"]),
            tp2_price=float(row["tp2_price"]),
            risk_per_unit=float(row["risk_per_unit"]),
            tp1_hit=bool(row["tp1_hit"]),
            tp2_hit=bool(row["tp2_hit"]),
            opened_at=datetime.fromisoformat(row["opened_ts"]),
        )

    def update_stop_loss(self, trade_id: int, stop_loss: float) -> None:
        self.conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (stop_loss, trade_id))
        self.conn.commit()

    def mark_tp_hit(self, trade_id: int, tp_level: int) -> None:
        if tp_level == 1:
            self.conn.execute("UPDATE trades SET tp1_hit = 1 WHERE id = ?", (trade_id,))
        elif tp_level == 2:
            self.conn.execute("UPDATE trades SET tp2_hit = 1 WHERE id = ?", (trade_id,))
        else:
            raise ValueError("tp_level must be 1 or 2")
        self.conn.commit()

    def apply_exit(
        self,
        trade_id: int,
        exit_qty: float,
        exit_price: float,
        event_type: str,
        notes: str,
        fee: float = 0.0,
    ) -> dict[str, float | bool]:
        row = self.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if row is None:
            raise ValueError(f"Trade id {trade_id} not found")

        if row["status"] != "OPEN":
            return {"closed": True, "remaining_qty": 0.0, "pnl": 0.0, "r_multiple": 0.0}

        side = row["side"]
        remaining = float(row["remaining_qty"])
        entry_price = float(row["entry_price"])
        risk_per_unit = float(row["risk_per_unit"])

        qty = min(max(exit_qty, 0.0), remaining)
        if qty <= 0:
            return {"closed": False, "remaining_qty": remaining, "pnl": 0.0, "r_multiple": 0.0}

        if side == "LONG":
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        risk_amount_piece = max(1e-12, risk_per_unit * qty)
        r_multiple = pnl / risk_amount_piece

        new_remaining = max(0.0, remaining - qty)
        close_now = new_remaining <= 1e-12

        self.conn.execute(
            """
            UPDATE trades
            SET remaining_qty = ?,
                realized_pnl = realized_pnl + ?,
                realized_r = realized_r + ?,
                fees = fees + ?,
                status = CASE WHEN ? THEN 'CLOSED' ELSE status END,
                closed_ts = CASE WHEN ? THEN ? ELSE closed_ts END,
                close_reason = CASE WHEN ? THEN ? ELSE close_reason END
            WHERE id = ?
            """,
            (
                new_remaining,
                pnl,
                r_multiple,
                fee,
                int(close_now),
                int(close_now),
                datetime.utcnow().isoformat(timespec="seconds"),
                int(close_now),
                notes,
                trade_id,
            ),
        )
        self._insert_trade_event(
            trade_id=trade_id,
            event_type=event_type,
            qty=qty,
            price=exit_price,
            pnl=pnl,
            r_multiple=r_multiple,
            fee=fee,
            notes=notes,
        )
        self.conn.commit()

        return {
            "closed": close_now,
            "remaining_qty": new_remaining,
            "pnl": pnl,
            "r_multiple": r_multiple,
        }

    def force_close_open_trade(self, symbol: str, exit_price: float, reason: str) -> None:
        pos = self.get_open_trade(symbol)
        if pos is None:
            return
        self.apply_exit(
            trade_id=pos.trade_id,
            exit_qty=pos.remaining_qty,
            exit_price=exit_price,
            event_type="FORCE_CLOSE",
            notes=reason,
        )

    def get_trade_snapshot(self, trade_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_total_realized_pnl(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS total FROM trades WHERE status = 'CLOSED'"
        ).fetchone()
        return float(row["total"])

    def get_daily_realized_pnl(self, utc_day_start: datetime) -> float:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0) AS total
            FROM trades
            WHERE status = 'CLOSED' AND closed_ts >= ?
            """,
            (utc_day_start.isoformat(timespec="seconds"),),
        ).fetchone()
        return float(row["total"])

    def get_weekly_realized_pnl(self, utc_week_start: datetime) -> float:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0) AS total
            FROM trades
            WHERE status = 'CLOSED' AND closed_ts >= ?
            """,
            (utc_week_start.isoformat(timespec="seconds"),),
        ).fetchone()
        return float(row["total"])

    def get_consecutive_losses(self, symbol: str, max_scan: int = 50) -> int:
        rows = self.conn.execute(
            """
            SELECT realized_pnl
            FROM trades
            WHERE symbol = ? AND status = 'CLOSED'
            ORDER BY id DESC
            LIMIT ?
            """,
            (symbol, max_scan),
        ).fetchall()

        losses = 0
        for row in rows:
            if float(row["realized_pnl"]) < 0:
                losses += 1
            else:
                break
        return losses

    def get_trade_count_today(self, symbol: str, utc_day_start: datetime) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM trades
            WHERE symbol = ? AND opened_ts >= ?
            """,
            (symbol, utc_day_start.isoformat(timespec="seconds")),
        ).fetchone()
        return int(row["cnt"])

    def upsert_candles(self, symbol: str, interval: str, candles: pd.DataFrame) -> int:
        if candles.empty:
            return 0

        records = []
        for row in candles.itertuples(index=False):
            records.append(
                (
                    symbol,
                    interval,
                    int(row.ts),
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    float(row.volume),
                )
            )

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO candle_cache
            (symbol, interval, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()
        return len(records)

    def get_cached_candles(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        rows = self.conn.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM candle_cache
            WHERE symbol = ? AND interval = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (symbol, interval, limit),
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        records = [dict(row) for row in rows]
        return pd.DataFrame(records).sort_values("ts").reset_index(drop=True)

    def _insert_trade_event(
        self,
        trade_id: int,
        event_type: str,
        qty: float,
        price: float,
        pnl: float,
        r_multiple: float,
        fee: float,
        notes: str,
    ) -> None:
        ts = datetime.utcnow().isoformat(timespec="seconds")
        self.conn.execute(
            """
            INSERT INTO trade_events (trade_id, ts, event_type, qty, price, pnl, r_multiple, fee, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (trade_id, ts, event_type, qty, price, pnl, r_multiple, fee, notes),
        )

    def get_paper_report(self, starting_capital_usdt: float) -> dict[str, float | int]:
        rows = self.conn.execute(
            """
            SELECT ts, message, data
            FROM events
            ORDER BY id ASC
            """
        ).fetchall()

        total_candidates = 0
        accepted = 0
        rejected = 0
        closed = 0
        wins = 0
        sum_r = 0.0
        latest_equity = float(starting_capital_usdt)

        for row in rows:
            try:
                payload = json.loads(row["data"] or "{}")
            except json.JSONDecodeError:
                continue

            if not payload.get("paper_mode"):
                continue

            event_name = str(payload.get("event", "")).strip().lower()
            if event_name == "paper_candidate":
                total_candidates += 1
            elif event_name == "paper_decision":
                decision = str(payload.get("decision", "")).strip().lower()
                if decision == "accepted":
                    accepted += 1
                elif decision == "rejected":
                    rejected += 1
            elif event_name == "paper_trade_closed":
                closed += 1
                r_val = float(payload.get("realized_r", 0.0))
                sum_r += r_val
                if r_val > 0:
                    wins += 1
                if payload.get("paper_equity") is not None:
                    latest_equity = float(payload.get("paper_equity"))

        avg_r = (sum_r / closed) if closed > 0 else 0.0
        win_rate = (wins / closed) if closed > 0 else 0.0

        return {
            "total_candidates": total_candidates,
            "accepted_trades": accepted,
            "rejected_trades": rejected,
            "closed_trades": closed,
            "win_rate": float(win_rate),
            "average_r": float(avg_r),
            "cumulative_r": float(sum_r),
            "paper_equity": float(latest_equity),
        }

    def close(self) -> None:
        self.conn.close()
