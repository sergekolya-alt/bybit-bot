from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import BotConfig
from .db import Database
from .exchange import BybitSpotClient
from .runtime_status import StatusStore



def collect_health_snapshot(cfg: BotConfig) -> dict[str, Any]:
    db = Database(cfg.db_path)
    try:
        status_store = StatusStore(cfg.status_file)
        status = status_store.load()

        day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        pnl = db.get_pnl_summary(day_start)
        open_pos = db.get_open_position(cfg.symbol)

        last_price = status.get("last_price")
        if last_price is None:
            try:
                last_price = BybitSpotClient(cfg.api_key, cfg.api_secret, cfg.bybit_testnet).get_last_price(
                    cfg.symbol
                )
            except Exception:  # noqa: BLE001
                last_price = None

        open_position = None
        if open_pos is not None:
            open_pnl = None
            if isinstance(last_price, (float, int)):
                open_pnl = (float(last_price) - open_pos.entry_price) * open_pos.qty

            open_position = {
                "trade_id": open_pos.trade_id,
                "symbol": open_pos.symbol,
                "qty": open_pos.qty,
                "entry_price": open_pos.entry_price,
                "stop_loss": open_pos.stop_loss,
                "take_profit": open_pos.take_profit,
                "opened_at": open_pos.opened_at.isoformat(timespec="seconds"),
                "open_pnl": open_pnl,
            }

        return {
            "bot_status": status.get("state", "unknown"),
            "last_signal_time": status.get("last_signal_time"),
            "last_signal": status.get("last_signal"),
            "last_tick_time": status.get("last_tick_time"),
            "last_price": last_price,
            "last_price_source": status.get("last_price_source"),
            "ws_connected": status.get("ws_connected"),
            "halted_reason": status.get("halted_reason", ""),
            "pnl_summary": pnl,
            "open_position": open_position,
        }
    finally:
        db.close()
