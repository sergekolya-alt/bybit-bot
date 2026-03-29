# BTCUSDT Bybit USDT Perpetual Futures Bot

Aggressive breakout momentum bot for **BTCUSDT USDT perpetual futures** using Bybit V5 API.

Key runtime constraints:
- dry-run by default
- testnet/mainnet switch
- REST candles only
- one position max
- no martingale / no averaging / no grid / no hedge mode

## Strategy (Rule-Based)

Timeframes:
- Signal: 15m
- Trend filter: 1h

Indicators:
- 15m: EMA20, EMA50, ATR(14), Donchian(20), Volume SMA20
- 1h: EMA50, EMA200, ADX(14)

Long entry:
1. 1h EMA50 > EMA200
2. 1h ADX > 22
3. 15m high >= Donchian upper(20)
4. 15m EMA20 > EMA50
5. 15m volume > SMA20(volume) * 1.0

Short entry:
1. 1h EMA50 < EMA200
2. 1h ADX > 22
3. 15m low <= Donchian lower(20)
4. 15m EMA20 < EMA50
5. 15m volume > SMA20(volume) * 1.0

Exits:
- Initial SL = 1.8 * ATR
- TP1 = 1.5R on 30%
- TP2 = 3R on 30%
- Remaining 40% trailing stop (recent 3 closed 15m candles)
- Opposite signal closes remainder

## AI Signal-Quality Filter

The AI model is an **entry filter only**:
1. Rule-based strategy generates candidate signal
2. AI model scores candidate
3. Trade is allowed only if `score >= AI_SCORE_THRESHOLD`

No AI control over stop/TP/trailing/risk exits.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Commands

Run bot once:
```bash
python main.py run-bot --once
```

Run bot continuously:
```bash
python main.py run-bot
```

Baseline backtest:
```bash
python main.py backtest --limit 3000
```

Backtest with AI filter:
```bash
python main.py backtest --limit 3000 --use-ai-filter
```

Backtest comparison (baseline vs AI filter):
```bash
python main.py backtest --limit 3000 --compare-ai-filter
```

Generate AI training dataset:
```bash
python main.py generate-dataset --limit 3000 --output ai_signal_dataset.csv
```

Train AI model:
```bash
python main.py train-ai --dataset ai_signal_dataset.csv --model-output models/ai_signal_filter.json
```

## AI Dataset Schema

Each candidate-signal row includes:
- `timestamp`, `symbol`, `side`, `entry_price`, `stop_distance`
- all model features (15m, 1h, context)
- `target` where:
  - `1` if `+1R` is hit before `-1R` after signal
  - `0` otherwise

## Runtime AI Config

In `.env`:
- `AI_FILTER_ENABLED=false`
- `AI_MODEL_PATH=models/ai_signal_filter.json`
- `AI_SCORE_THRESHOLD=0.55`

Enable AI filter in runtime by setting:
- `AI_FILTER_ENABLED=true`
