# --- src/feature_time.py ---
from __future__ import annotations

import pandas as pd
import numpy as np

NY_TZ = "America/New_York"

def _to_time(s: str) -> pd.Timestamp:
    # returns a time component we can combine with a date
    return pd.to_datetime(s).time()

def _next_monday(date: pd.Timestamp) -> pd.Timestamp:
    # date is date-like (no tz). Saturday=5, Sunday=6
    wd = date.weekday()
    if wd == 5:   # Saturday -> +2 days
        return date + pd.Timedelta(days=2)
    if wd == 6:   # Sunday   -> +1 day
        return date + pd.Timedelta(days=1)
    return date

def add_ny_session_markers(
    df: pd.DataFrame,
    *,
    dt_col: str = "Gmt time",
    market_open_local: str = "09:30",
    market_close_local: str = "16:00",
    warmup_minutes: int = 30,
    roll_weekends: bool = True,   # <— NEW: remap Sat/Sun to Monday
) -> pd.DataFrame:
    """
    Adds:
      - dt_utc (tz-aware UTC)
      - dt_ny (tz-aware America/New_York)
      - date_us_open (date for the US session day; Mon–Fri if roll_weekends=True)
      - ny_open_ny / ny_close_ny / ny_warmup_end_ny (tz-aware NY)
      - ny_open_utc / ny_close_utc / ny_warmup_end_utc (tz-aware UTC)

    Behavior:
      - If roll_weekends=True: any row whose NY-local date falls on Sat/Sun is
        assigned to the NEXT Monday for its date_us_open and markers, so you
        never “create” a Sunday US day. Sunday evening FX trade will belong to
        Monday’s session, which fixes ATR/VWAP context and update timing.
    """
    x = df.copy()

    # 1) Core timestamps
    if dt_col not in x.columns:
        raise KeyError(f"Column '{dt_col}' not found in dataframe.")
    x["dt_utc"] = pd.to_datetime(x[dt_col], utc=True)
    x["dt_ny"]  = x["dt_utc"].dt.tz_convert(NY_TZ)

    # 2) Base US session date (from NY-local calendar day)
    ny_date = x["dt_ny"].dt.date
    ny_date = pd.to_datetime(ny_date)  # normalizes to Timestamp (naive date)
    if roll_weekends:
        rolled = ny_date.map(_next_monday)
        x["date_us_open"] = rolled.dt.date
    else:
        x["date_us_open"] = ny_date.dt.date

    # 3) Build NY-local markers from date_us_open
    t_open  = _to_time(market_open_local)
    t_close = _to_time(market_close_local)

    # Combine date_us_open with times in NY tz (DST-safe)
    # Create a naive datetime first, then localize to NY tz.
    open_naive  = pd.to_datetime(x["date_us_open"].astype(str) + " " + str(t_open))
    close_naive = pd.to_datetime(x["date_us_open"].astype(str) + " " + str(t_close))

    x["ny_open_ny"]        = open_naive.dt.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
    x["ny_close_ny"]       = close_naive.dt.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
    x["ny_warmup_end_ny"]  = x["ny_open_ny"] + pd.Timedelta(minutes=warmup_minutes)

    # 4) UTC versions
    x["ny_open_utc"]       = x["ny_open_ny"].dt.tz_convert("UTC")
    x["ny_close_utc"]      = x["ny_close_ny"].dt.tz_convert("UTC")
    x["ny_warmup_end_utc"] = x["ny_warmup_end_ny"].dt.tz_convert("UTC")

    return x
