from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from pybit.unified_trading import HTTP


@dataclass(frozen=True)
class InstrumentRules:
    min_qty: float
    qty_step: float
    tick_size: float
    min_notional: float


@dataclass(frozen=True)
class ExchangePosition:
    side: str  # LONG or SHORT
    qty: float
    entry_price: float


class BybitClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool) -> None:
        self.session = HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)
        self._rules_cache: dict[str, InstrumentRules] = {}

    @staticmethod
    def _expect_ok(resp: dict, context: str) -> dict:
        if resp.get("retCode") != 0:
            raise RuntimeError(f"Bybit {context} error: {resp}")
        return resp

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        params: dict[str, object] = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["start"] = start_ms
        if end_ms is not None:
            params["end"] = end_ms

        resp = self._expect_ok(self.session.get_kline(**params), "kline")

        records = []
        for row in resp["result"]["list"]:
            records.append(
                {
                    "ts": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )

        return pd.DataFrame(records).sort_values("ts").reset_index(drop=True)

    def get_last_price(self, symbol: str) -> float:
        resp = self._expect_ok(self.session.get_tickers(category="linear", symbol=symbol), "ticker")
        return float(resp["result"]["list"][0]["lastPrice"])

    def get_usdt_balance(self) -> float:
        resp = self._expect_ok(
            self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT"),
            "wallet_balance",
        )
        for acct in resp["result"]["list"]:
            for coin in acct.get("coin", []):
                if coin.get("coin") == "USDT":
                    return float(coin.get("walletBalance", "0"))
        return 0.0

    def get_instrument_rules(self, symbol: str) -> InstrumentRules:
        cached = self._rules_cache.get(symbol)
        if cached is not None:
            return cached

        resp = self._expect_ok(
            self.session.get_instruments_info(category="linear", symbol=symbol),
            "instruments_info",
        )
        info = resp["result"]["list"][0]

        lot = info.get("lotSizeFilter", {})
        price = info.get("priceFilter", {})

        parsed = InstrumentRules(
            min_qty=float(lot.get("minOrderQty", "0.001")),
            qty_step=float(lot.get("qtyStep", "0.001")),
            tick_size=float(price.get("tickSize", "0.1")),
            min_notional=float(lot.get("minNotionalValue", "0")),
        )
        self._rules_cache[symbol] = parsed
        return parsed

    def set_leverage(self, symbol: str, leverage: int) -> None:
        resp = self.session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage),
        )
        code = int(resp.get("retCode", -1))
        # 0 success, 110043 usually means leverage already set.
        if code not in {0, 110043}:
            raise RuntimeError(f"Bybit set_leverage error: {resp}")

    def get_open_position(self, symbol: str) -> ExchangePosition | None:
        resp = self._expect_ok(self.session.get_positions(category="linear", symbol=symbol), "positions")
        for row in resp["result"]["list"]:
            size = float(row.get("size", "0") or 0)
            if size <= 0:
                continue
            side = row.get("side", "")
            if side == "Buy":
                parsed_side = "LONG"
            elif side == "Sell":
                parsed_side = "SHORT"
            else:
                continue
            return ExchangePosition(
                side=parsed_side,
                qty=size,
                entry_price=float(row.get("avgPrice", "0") or 0),
            )
        return None

    def place_market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> str:
        if side not in {"Buy", "Sell"}:
            raise ValueError("side must be Buy or Sell")

        resp = self._expect_ok(
            self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                reduceOnly=reduce_only,
                positionIdx=0,
                timeInForce="IOC",
            ),
            "place_order",
        )
        return str(resp["result"]["orderId"])

    @staticmethod
    def round_qty_down(qty: float, step: float) -> float:
        if step <= 0:
            return qty
        units = math.floor(qty / step)
        rounded = units * step
        decimals = max(0, len(str(step).split(".")[-1].rstrip("0")))
        return round(rounded, decimals)

    @staticmethod
    def round_price(price: float, tick_size: float, mode: str = "nearest") -> float:
        if tick_size <= 0:
            return price

        units = price / tick_size
        if mode == "down":
            adj = math.floor(units)
        elif mode == "up":
            adj = math.ceil(units)
        else:
            adj = round(units)

        decimals = max(0, len(str(tick_size).split(".")[-1].rstrip("0")))
        return round(adj * tick_size, decimals)


# Backward alias to avoid touching unrelated imports.
BybitSpotClient = BybitClient
