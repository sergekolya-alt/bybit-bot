from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .config import BotConfig
from .strategy import BreakoutMomentumStrategy
from .strategy.indicators import adx, atr, donchian_lower, donchian_upper, ema, sma

FEATURE_COLUMNS = [
    "close_to_ema20_pct",
    "close_to_ema50_pct",
    "ema20_to_ema50_pct",
    "atr_pct",
    "donchian_breakout_distance_pct",
    "candle_body_pct_of_range",
    "upper_wick_pct_of_range",
    "lower_wick_pct_of_range",
    "volume_to_sma20_ratio",
    "return_1",
    "return_3",
    "return_6",
    "trend_up_flag",
    "ema50_to_ema200_pct",
    "adx_1h",
    "return_1h_3",
    "return_1h_6",
    "side_flag",
    "hour_of_day",
]


@dataclass(frozen=True)
class AIScoreDecision:
    allow: bool
    score: float


class AISignalFilter:
    def __init__(self, enabled: bool, model_path: str, threshold: float) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self.threshold = threshold
        self._model: Any | None = None
        self._model_kind = "none"
        self._feature_columns = list(FEATURE_COLUMNS)

        if self.enabled:
            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(f"AI model file not found: {self.model_path}")

            if path.suffix.lower() in {".joblib", ".pkl"}:
                import joblib

                loaded = joblib.load(path)
                if isinstance(loaded, dict) and "model" in loaded:
                    self._model = loaded["model"]
                    cols = loaded.get("feature_columns")
                    if isinstance(cols, list) and cols:
                        self._feature_columns = [str(c) for c in cols]
                else:
                    self._model = loaded
                self._model_kind = "sklearn"
            else:
                from xgboost import XGBClassifier

                model = XGBClassifier()
                model.load_model(str(path))
                self._model = model
                self._model_kind = "xgboost"

    def score(self, features: dict[str, float]) -> float:
        if not self.enabled:
            return 1.0
        if self._model is None:
            raise RuntimeError("AI filter is enabled but model is not loaded")

        missing = [col for col in self._feature_columns if col not in features]
        if missing:
            raise ValueError(f"Missing runtime features for model inference: {missing}")

        row = {col: float(features[col]) for col in self._feature_columns}
        x = pd.DataFrame([row])
        prob = float(self._model.predict_proba(x)[0][1])
        return prob

    def decide(self, features: dict[str, float]) -> AIScoreDecision:
        score = self.score(features)
        return AIScoreDecision(allow=score >= self.threshold, score=score)

    def validate_runtime_schema(self, runtime_feature_columns: list[str] | None = None) -> None:
        runtime = list(runtime_feature_columns or FEATURE_COLUMNS)
        if self._feature_columns != runtime:
            raise ValueError(
                (
                    "Model feature schema mismatch: "
                    f"model={self._feature_columns} runtime={runtime}"
                )
            )


def _with_signal_indicators(candles_15m: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    df = candles_15m.copy().sort_values("ts").reset_index(drop=True)
    df["ema20"] = ema(df["close"], cfg.signal_ema_fast)
    df["ema50"] = ema(df["close"], cfg.signal_ema_slow)
    df["atr"] = atr(df["high"], df["low"], df["close"], cfg.atr_period)
    df["donchian_upper"] = donchian_upper(df["high"], cfg.donchian_period)
    df["donchian_lower"] = donchian_lower(df["low"], cfg.donchian_period)
    df["vol_sma20"] = sma(df["volume"], cfg.volume_sma_period)
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    return df


def _with_trend_indicators(candles_1h: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    df = candles_1h.copy().sort_values("ts").reset_index(drop=True)
    df["ema50"] = ema(df["close"], cfg.trend_ema_fast)
    df["ema200"] = ema(df["close"], cfg.trend_ema_slow)
    df["adx"] = adx(df["high"], df["low"], df["close"], cfg.adx_period)
    df["return_1h_3"] = df["close"].pct_change(3)
    df["return_1h_6"] = df["close"].pct_change(6)
    return df


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


def _build_feature_row(last15: pd.Series, last1h: pd.Series, side: str, ts_ms: int) -> dict[str, float]:
    close = float(last15["close"])
    open_ = float(last15["open"])
    high = float(last15["high"])
    low = float(last15["low"])

    candle_range = max(1e-12, high - low)
    body = abs(close - open_)
    upper_wick = max(0.0, high - max(open_, close))
    lower_wick = max(0.0, min(open_, close) - low)

    if side == "LONG":
        breakout_dist = _safe_ratio(float(last15["high"]) - float(last15["donchian_upper"]), close)
        side_flag = 1.0
    else:
        breakout_dist = _safe_ratio(float(last15["donchian_lower"]) - float(last15["low"]), close)
        side_flag = 0.0

    ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

    out = {
        "close_to_ema20_pct": _safe_ratio(close - float(last15["ema20"]), float(last15["ema20"])),
        "close_to_ema50_pct": _safe_ratio(close - float(last15["ema50"]), float(last15["ema50"])),
        "ema20_to_ema50_pct": _safe_ratio(float(last15["ema20"]) - float(last15["ema50"]), float(last15["ema50"])),
        "atr_pct": _safe_ratio(float(last15["atr"]), close),
        "donchian_breakout_distance_pct": breakout_dist,
        "candle_body_pct_of_range": body / candle_range,
        "upper_wick_pct_of_range": upper_wick / candle_range,
        "lower_wick_pct_of_range": lower_wick / candle_range,
        "volume_to_sma20_ratio": _safe_ratio(float(last15["volume"]), float(last15["vol_sma20"])),
        "return_1": float(last15["return_1"]),
        "return_3": float(last15["return_3"]),
        "return_6": float(last15["return_6"]),
        "trend_up_flag": 1.0 if float(last1h["ema50"]) > float(last1h["ema200"]) else 0.0,
        "ema50_to_ema200_pct": _safe_ratio(float(last1h["ema50"]) - float(last1h["ema200"]), float(last1h["ema200"])),
        "adx_1h": float(last1h["adx"]),
        "return_1h_3": float(last1h["return_1h_3"]),
        "return_1h_6": float(last1h["return_1h_6"]),
        "side_flag": side_flag,
        "hour_of_day": float(ts_dt.hour),
    }

    return {k: (0.0 if pd.isna(v) else float(v)) for k, v in out.items()}


def build_runtime_feature_vector(
    candles_15m: pd.DataFrame,
    candles_1h: pd.DataFrame,
    cfg: BotConfig,
    side: str,
) -> dict[str, float] | None:
    min_signal = max(cfg.signal_ema_slow, cfg.donchian_period, cfg.volume_sma_period) + 3
    min_trend = max(cfg.trend_ema_slow, cfg.adx_period) + 3
    if len(candles_15m) < min_signal or len(candles_1h) < min_trend:
        return None

    s15 = _with_signal_indicators(candles_15m, cfg)
    t1h = _with_trend_indicators(candles_1h, cfg)

    last15 = s15.iloc[-1]
    last1h = t1h.iloc[-1]
    ts_ms = int(last15["ts"])
    return _build_feature_row(last15, last1h, side, ts_ms)


def _trade_outcome(
    candles_15m: pd.DataFrame,
    signal_idx: int,
    side: str,
    entry_price: float,
    stop_distance: float,
) -> tuple[int, float, str, int, int]:
    signal_ts = int(candles_15m.iloc[signal_idx]["ts"])
    if stop_distance <= 0:
        return 0, 0.0, "invalid_stop", signal_ts, 0

    if side == "LONG":
        take_price = entry_price + stop_distance
        stop_price = entry_price - stop_distance
    else:
        take_price = entry_price - stop_distance
        stop_price = entry_price + stop_distance

    for j in range(signal_idx + 1, len(candles_15m)):
        row = candles_15m.iloc[j]
        high = float(row["high"])
        low = float(row["low"])

        if side == "LONG":
            hit_tp = high >= take_price
            hit_sl = low <= stop_price
        else:
            hit_tp = low <= take_price
            hit_sl = high >= stop_price

        exit_ts = int(row["ts"])
        duration_bars = j - signal_idx
        if hit_tp and hit_sl:
            return 0, -1.0, "both_hit_same_candle", exit_ts, duration_bars
        if hit_tp:
            return 1, 1.0, "tp_1r", exit_ts, duration_bars
        if hit_sl:
            return 0, -1.0, "sl_1r", exit_ts, duration_bars

    last_close = float(candles_15m.iloc[-1]["close"])
    exit_ts = int(candles_15m.iloc[-1]["ts"])
    duration_bars = max(0, len(candles_15m) - 1 - signal_idx)
    if side == "LONG":
        realized_r = _safe_ratio(last_close - entry_price, stop_distance)
    else:
        realized_r = _safe_ratio(entry_price - last_close, stop_distance)
    return 0, float(realized_r), "no_1r_hit_before_series_end", exit_ts, duration_bars


def build_signal_dataset(
    candles_15m: pd.DataFrame,
    candles_1h: pd.DataFrame,
    cfg: BotConfig,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    c15_raw = candles_15m.copy().sort_values("ts").reset_index(drop=True)
    c1h_raw = candles_1h.copy().sort_values("ts").reset_index(drop=True)
    s15 = _with_signal_indicators(c15_raw, cfg)
    t1h = _with_trend_indicators(c1h_raw, cfg)

    strategy = BreakoutMomentumStrategy(cfg)
    min_signal = max(cfg.signal_ema_slow, cfg.donchian_period, cfg.volume_sma_period) + 3
    min_trend = max(cfg.trend_ema_slow, cfg.adx_period) + 3

    rows: list[dict[str, Any]] = []
    candidate_signals = 0
    for i in range(min_signal, len(c15_raw) - 1):
        ts = int(c15_raw.iloc[i]["ts"])
        idx1 = int(c1h_raw["ts"].searchsorted(ts, side="right") - 1)
        if idx1 + 1 < min_trend:
            continue

        hist15 = c15_raw.iloc[: i + 1]
        hist1h = c1h_raw.iloc[: idx1 + 1]
        signal = strategy.entry_signal(hist15, hist1h)
        if signal.side == "NONE":
            continue
        candidate_signals += 1

        entry_price = float(c15_raw.iloc[i]["close"])
        stop_distance = cfg.atr_stop_mult * float(s15.iloc[i]["atr"])
        if stop_distance <= 0:
            continue

        features = _build_feature_row(s15.iloc[i], t1h.iloc[idx1], signal.side, ts)
        target, realized_r, outcome_type, exit_ts, duration_bars = _trade_outcome(
            c15_raw, i, signal.side, entry_price, stop_distance
        )
        realized_return_pct = _safe_ratio(realized_r * stop_distance, entry_price)

        row: dict[str, Any] = {
            "timestamp": ts,
            "exit_timestamp": exit_ts,
            "duration_bars_15m": int(duration_bars),
            "duration_minutes": int(duration_bars * 15),
            "symbol": cfg.symbol,
            "side": signal.side,
            "entry_price": entry_price,
            "stop_distance": stop_distance,
            "target": int(target),
            "realized_r": float(realized_r),
            "realized_return_pct": float(realized_return_pct),
            "outcome_type": outcome_type,
        }
        row.update(features)
        rows.append(row)

    columns = [
        "timestamp",
        "exit_timestamp",
        "duration_bars_15m",
        "duration_minutes",
        "symbol",
        "side",
        "entry_price",
        "stop_distance",
        *FEATURE_COLUMNS,
        "target",
        "realized_r",
        "realized_return_pct",
        "outcome_type",
    ]
    ds = pd.DataFrame(rows, columns=columns)
    if return_stats:
        return ds, {"candidate_signals": candidate_signals}
    return ds


def train_ai_model(dataset_csv: str, model_output: str, split_ratio: float = 0.8) -> dict[str, Any]:
    from xgboost import XGBClassifier

    df = pd.read_csv(dataset_csv).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset is empty")

    if "target" not in df.columns:
        raise ValueError("Dataset must contain 'target' column")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing feature column in dataset: {col}")

    split_idx = int(len(df) * split_ratio)
    split_idx = max(1, min(split_idx, len(df) - 1))

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"].astype(int)
    x_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["target"].astype(int)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)

    roc_auc: float | None = None
    if y_val.nunique() > 1:
        roc_auc = float(roc_auc_score(y_val, val_prob))

    precision = float(precision_score(y_val, val_pred, zero_division=0))
    recall = float(recall_score(y_val, val_pred, zero_division=0))

    importances = list(model.feature_importances_)
    ranked = sorted(
        [{"feature": FEATURE_COLUMNS[i], "importance": float(importances[i])} for i in range(len(FEATURE_COLUMNS))],
        key=lambda item: item["importance"],
        reverse=True,
    )

    out_path = Path(model_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))

    return {
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "feature_importance": ranked[:10],
        "model_path": str(out_path),
    }


def train_rf_model(dataset_csv: str, model_output: str, split_ratio: float = 0.8) -> dict[str, Any]:
    import joblib

    df = pd.read_csv(dataset_csv).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset is empty")

    if "target" not in df.columns:
        raise ValueError("Dataset must contain 'target' column")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing feature column in dataset: {col}")

    split_idx = int(len(df) * split_ratio)
    split_idx = max(1, min(split_idx, len(df) - 1))

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"].astype(int)
    x_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["target"].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)

    roc_auc: float | None = None
    if y_val.nunique() > 1:
        roc_auc = float(roc_auc_score(y_val, val_prob))

    precision = float(precision_score(y_val, val_pred, zero_division=0))
    recall = float(recall_score(y_val, val_pred, zero_division=0))

    importances = list(model.feature_importances_)
    ranked = sorted(
        [{"feature": FEATURE_COLUMNS[i], "importance": float(importances[i])} for i in range(len(FEATURE_COLUMNS))],
        key=lambda item: item["importance"],
        reverse=True,
    )

    artifact = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_family": "RandomForestClassifier",
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }

    out_path = Path(model_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)

    return {
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "feature_importance": ranked[:10],
        "model_path": str(out_path),
        "feature_columns": list(FEATURE_COLUMNS),
        "model_family": "RandomForestClassifier",
    }
