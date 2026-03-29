from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import websocket


class BybitPriceStream:
    def __init__(self, symbol: str, testnet: bool, reconnect_delay_sec: int = 5) -> None:
        self.symbol = symbol
        self.testnet = testnet
        self.reconnect_delay_sec = reconnect_delay_sec

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws: websocket.WebSocketApp | None = None

        self._latest_price: float | None = None
        self._latest_ts: float | None = None
        self._connected = False
        self._last_error = ""

        self._logger = logging.getLogger(__name__)

    @property
    def ws_url(self) -> str:
        if self.testnet:
            return "wss://stream-testnet.bybit.com/v5/public/spot"
        return "wss://stream.bybit.com/v5/public/spot"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="bybit-price-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def latest_price(self, max_age_sec: int) -> float | None:
        now = time.time()
        with self._lock:
            if self._latest_price is None or self._latest_ts is None:
                return None
            if now - self._latest_ts > max_age_sec:
                return None
            return self._latest_price

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self._connected,
                "last_price": self._latest_price,
                "last_price_ts": self._latest_ts,
                "last_error": self._last_error,
            }

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            self._connected = False
            self._ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._logger.info("Starting price stream", extra={"ctx": {"url": self.ws_url}})
            self._ws.run_forever(ping_interval=20, ping_timeout=10)

            if self._stop.is_set():
                break
            time.sleep(self.reconnect_delay_sec)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        sub = {"op": "subscribe", "args": [f"tickers.{self.symbol}"]}
        ws.send(json.dumps(sub, ensure_ascii=True))
        with self._lock:
            self._connected = True
            self._last_error = ""

    def _on_message(self, _ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:  # noqa: BLE001
            return

        topic = msg.get("topic", "")
        if topic != f"tickers.{self.symbol}":
            return

        data = msg.get("data") or {}
        if not isinstance(data, dict):
            return

        last_price = data.get("lastPrice")
        if last_price is None:
            return

        try:
            parsed = float(last_price)
        except (TypeError, ValueError):
            return

        with self._lock:
            self._latest_price = parsed
            self._latest_ts = time.time()

    def _on_error(self, _ws: websocket.WebSocketApp, error: Any) -> None:
        with self._lock:
            self._connected = False
            self._last_error = str(error)
        self._logger.warning("Price stream error", extra={"ctx": {"error": str(error)}})

    def _on_close(
        self,
        _ws: websocket.WebSocketApp,
        close_status_code: int | None,
        close_msg: str | None,
    ) -> None:
        with self._lock:
            self._connected = False
        self._logger.info(
            "Price stream closed",
            extra={"ctx": {"code": close_status_code, "message": close_msg}},
        )
