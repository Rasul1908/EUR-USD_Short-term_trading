""\
# Backtest and Machine Learning Framework — Enhanced by Feature Engineering for Volatility-Driven Trading Signals

This repository contains a conservative, session-aware **intraday backtesting** and **logistic-regression filtering** pipeline for EUR/USD minute data. It centers on **feature engineering** tied to the U.S. trading session (NY 09:30–16:00), a **Fair Value (FV) zone** derived from the Initial Balance (first 30 minutes), and **L1 extensions** scaled by a **pre-US volatility score**. The ML layer gates trades rather than replacing the rules, improving selectivity.

---

## Key Features

- **Sessionization (NY time):** Robust markers for `ny_open`, `ny_close`, and `ny_warmup_end` with weekend roll (Sun trades belong to Monday’s session).
- **Volatility regime tagging:** Pre-US range vs. 14-day ATR → `vol_score` (clipped to [0.7, 1.3]) and optional `is_volatile` flag.
- **Level engineering:** FV zone from the IB; optional VWAP-blended midpoint; **L1** projections with modes for volatility scaling (`up_only`, `both`, `none`).
- **Anti-look-ahead:** “Active” levels use prior day until 10:00 NY; no intrabar fantasy fills; pessimistic TP/SL tie-breaks.
- **ML trade filter (logistic):** Uses engineered session features to rank trades and execute only above a probability threshold chosen on train/validation.
- **Reproducibility:** Parameterized pipeline; deterministic seed; exports trades, KPIs, and figures.

---

## Project Structure