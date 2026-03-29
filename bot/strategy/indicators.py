from __future__ import annotations

import pandas as pd



def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()



def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()



def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()



def donchian_upper(high: pd.Series, period: int = 20) -> pd.Series:
    # Use only closed candles before current row to avoid lookahead.
    return high.shift(1).rolling(period).max()



def donchian_lower(low: pd.Series, period: int = 20) -> pd.Series:
    # Use only closed candles before current row to avoid lookahead.
    return low.shift(1).rolling(period).min()



def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)

    atr_smoothed = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_smoothed.replace(0, float("nan")))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_smoothed.replace(0, float("nan")))

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))).fillna(0.0)
    out = dx.ewm(alpha=1 / period, adjust=False).mean()
    return out.fillna(0)
