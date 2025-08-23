# --- TIME FEATURES (src/features_time.py) ---
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import time

# ============================================================
# Parsing & helpers
# ============================================================

def _parse_datetime(df: pd.DataFrame, ts_col: str = "Gmt time") -> pd.DataFrame:
    """
    Parse timestamps like '16.08.2023 00:00:00.000' as UTC (tz-aware).
    Falls back to pandas inference if strict parse fails on any rows.
    """
    out = df.copy()
    dt = pd.to_datetime(out[ts_col], format="%d.%m.%Y %H:%M:%S.%f",
                        errors="coerce", utc=True)
    if dt.isna().any():
        dt2 = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
        dt = dt.fillna(dt2)
    out["datetime"] = dt
    if out["datetime"].isna().any():
        raise ValueError("Some timestamps could not be parsed; check input format.")
    return out

def _cyclical_encode(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    """Return sin/cos encodings for periodic integer values (e.g., hour 0..23)."""
    radians = 2 * np.pi * (series % period) / period
    return pd.DataFrame({
        f"{prefix}_sin": np.sin(radians),
        f"{prefix}_cos": np.cos(radians)
    }, index=series.index)

def _in_window_vec(local_time_series: pd.Series, start: time, end: time) -> pd.Series:
    """
    Vectorized membership for [start, end) in local time.
    Supports windows that cross midnight (e.g., 22:00 -> 06:00).
    """
    t = local_time_series
    if start <= end:
        return (t >= start) & (t < end)
    else:
        # wrap-around: late-night OR early-morning
        return (t >= start) | (t < end)

# ============================================================
# Core time features
# ============================================================

def add_time_core(df: pd.DataFrame, ts_col: str = "Gmt time") -> pd.DataFrame:
    """
    Adds: datetime (UTC tz-aware), date, hour (UTC), day_of_week (0=Mon..6=Sun), month (1..12).
    """
    out = _parse_datetime(df, ts_col=ts_col)
    out["date"] = out["datetime"].dt.date
    out["hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek  # 0..6
    out["month"] = out["datetime"].dt.month            # 1..12
    return out

def add_hour_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Adds hour_sin/hour_cos from UTC hour."""
    out = df.copy()
    if "hour" not in out.columns:
        raise ValueError("Run add_time_core() first so 'hour' exists.")
    enc = _cyclical_encode(out["hour"], period=24, prefix="hour")
    out = pd.concat([out, enc], axis=1)
    return out

def add_day_month_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - dow_sin/dow_cos (day_of_week cyclical, 0..6 with period=7)
      - month_sin/month_cos (month cyclical, 1..12 mapped to 0..11 with period=12)
    """
    out = df.copy()
    if "day_of_week" not in out.columns or "month" not in out.columns:
        raise ValueError("Run add_time_core() first so 'day_of_week' and 'month' exist.")
    dow_enc   = _cyclical_encode(out["day_of_week"], period=7, prefix="dow")
    month_enc = _cyclical_encode((out["month"] - 1) % 12, period=12, prefix="month")
    out = pd.concat([out, dow_enc, month_enc], axis=1)
    return out

# ============================================================
# DST-aware session labeling (FX desk or equities exchange)
# ============================================================

def add_session_features_dst(
    df: pd.DataFrame,
    market: str = "fx",  # "fx" or "equities"
    sydney_window: tuple[time, time] | None = None,
    tokyo_window:  tuple[time, time] | None = None,
    london_window: tuple[time, time] | None = None,
    ny_window:     tuple[time, time] | None = None,
    dominance: tuple[str, ...] = ("NewYork", "London", "Tokyo", "Sydney"),
) -> pd.DataFrame:
    """
    Label sessions using local market hours (DST-aware via tz_convert).
    Defaults:
      market="fx" (desk-style):
        Sydney 08:00–17:00, Tokyo 09:00–18:00, London 08:00–17:00, New York 08:00–17:00
      market="equities" (exchange hours):
        Sydney 10:00–16:00, Tokyo 09:00–15:00 (ignores lunch), London 08:00–16:30, New York 09:30–16:00
    You can override any window via the *_window args (local times).

    Adds:
      in_Sydney, in_Tokyo, in_London, in_NewYork  (Int8)
      overlap_Sydney_Tokyo, overlap_Tokyo_London, overlap_London_NewYork (Int8)
      session (dominant by priority order)
    """
    out = df.copy()
    if "datetime" not in out.columns:
        raise ValueError("Run add_time_core() first so 'datetime' exists (tz-aware).")

    # Defaults by market
    if market.lower() == "equities":
        defaults = {
            "sydney": (time(10,0), time(16,0)),
            "tokyo":  (time(9,0),  time(15,0)),   # lunch ignored
            "london": (time(8,0),  time(16,30)),
            "ny":     (time(9,30), time(16,0)),
        }
    else:  # "fx"
        defaults = {
            "sydney": (time(8,0),  time(17,0)),
            "tokyo":  (time(9,0),  time(18,0)),
            "london": (time(8,0),  time(17,0)),
            "ny":     (time(8,0),  time(17,0)),
        }

    sydney_window = sydney_window or defaults["sydney"]
    tokyo_window  = tokyo_window  or defaults["tokyo"]
    london_window = london_window or defaults["london"]
    ny_window     = ny_window     or defaults["ny"]

    # Local wall-clock times (DST-aware)
    syd_local = out["datetime"].dt.tz_convert("Australia/Sydney").dt.time
    tyo_local = out["datetime"].dt.tz_convert("Asia/Tokyo").dt.time
    lon_local = out["datetime"].dt.tz_convert("Europe/London").dt.time
    ny_local  = out["datetime"].dt.tz_convert("America/New_York").dt.time

    # Session flags
    out["in_Sydney"]  = _in_window_vec(syd_local, sydney_window[0], sydney_window[1]).astype("int8")
    out["in_Tokyo"]   = _in_window_vec(tyo_local, tokyo_window[0],  tokyo_window[1]).astype("int8")
    out["in_London"]  = _in_window_vec(lon_local, london_window[0], london_window[1]).astype("int8")
    out["in_NewYork"] = _in_window_vec(ny_local,  ny_window[0],     ny_window[1]).astype("int8")

    # Overlaps
    out["overlap_Sydney_Tokyo"]   = ((out["in_Sydney"]  == 1) & (out["in_Tokyo"]   == 1)).astype("int8")
    out["overlap_Tokyo_London"]   = ((out["in_Tokyo"]   == 1) & (out["in_London"]  == 1)).astype("int8")
    out["overlap_London_NewYork"] = ((out["in_London"]  == 1) & (out["in_NewYork"] == 1)).astype("int8")

    # Dominant session label by priority
    order = list(dominance) + ["Off"]
    m = pd.DataFrame({
        "Sydney":  out["in_Sydney"].astype(bool),
        "Tokyo":   out["in_Tokyo"].astype(bool),
        "London":  out["in_London"].astype(bool),
        "NewYork": out["in_NewYork"].astype(bool),
    }, index=out.index)

    def _pick(row) -> str:
        for name in order[:-1]:
            if row.get(name, False):
                return name
        return "Off"

    out["session"] = m.apply(_pick, axis=1)
    return out

def add_calendar_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: one-hot for session and day_of_week.
    Produces: session_[Sydney|Tokyo|London|NewYork|Off], dow_[0..6] (Mon..Sun).
    """
    out = df.copy()
    if "session" not in out.columns or "day_of_week" not in out.columns:
        raise ValueError("Run add_time_core() and add_session_features_dst() first.")
    sess_ohe = pd.get_dummies(out["session"], prefix="session")
    dow_ohe  = pd.get_dummies(out["day_of_week"], prefix="dow")
    out = pd.concat([out, sess_ohe, dow_ohe], axis=1)
    return out

# ============================================================
# Convenience wrapper and model-time feature pickers
# ============================================================

def add_time_features(
    df: pd.DataFrame,
    ts_col: str = "Gmt time",
    market: str = "fx",
    add_sessions: bool = True,
    add_cyclical: bool = False,
) -> pd.DataFrame:
    """
    Convenience:
      - parses datetime/date/hour/dow/month
      - (optional) session flags (DST-aware, market-specific windows)
      - (optional) cyclical encodings for hour/dow/month
    """
    out = add_time_core(df, ts_col=ts_col)
    if add_sessions:
        out = add_session_features_dst(out, market=market)
    if add_cyclical:
        out = add_hour_cyclical(out)
        out = add_day_month_cyclical(out)
    return out

def time_feature_names(mode: str = "tree") -> list[str]:
    """
    Returns recommended time-column names by model family.
      - 'tree': raw ints (hour, day_of_week, month) + session flags
      - 'nn'  : cyclical (hour/dow) + session flags
    """
    if mode == "nn":
        base = ["hour_sin","hour_cos","dow_sin","dow_cos"]
    else:
        base = ["hour","day_of_week","month"]
    # session flags are useful to both; add if present
    base += ["in_Sydney","in_Tokyo","in_London","in_NewYork",
             "overlap_Sydney_Tokyo","overlap_Tokyo_London","overlap_London_NewYork"]
    return base


# --- add to src/features_time.py ---
from datetime import time
import pandas as pd

def add_us_trading_day(df: pd.DataFrame,
                       *,
                       dt_col: str = "datetime",
                       anchor_ny: time = time(10, 0),
                       out_col: str = "date_us") -> pd.DataFrame:
    """
    Adds `date_us`: a US-anchored trading day that starts at `anchor_ny`
    (default 10:00 America/New_York) and runs 24h to next anchor.
    """
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], utc=True, errors="coerce")
    ny = out[dt_col].dt.tz_convert("America/New_York")
    ny_date = ny.dt.date
    ny_time = ny.dt.time

    # if local time < anchor -> belongs to previous US day
    us_date = pd.Series(ny_date, index=out.index, dtype="object")
    mask = ny_time < anchor_ny
    us_date.loc[mask] = (ny[mask] - pd.Timedelta(days=1)).dt.date
    out[out_col] = us_date
    return out

# === NEW: shared NY session markers (UTC + NY) ===
from zoneinfo import ZoneInfo

def add_ny_session_markers(
    df: pd.DataFrame,
    *,
    dt_col: str = "datetime",            # tz-aware UTC column
    tz_market: str = "America/New_York",
    market_open_local: str = "09:30",
    market_close_local: str = "16:00",
    warmup_minutes: int = 30
) -> pd.DataFrame:
    """
    Standardizes all NY session fields:
      - dt_utc (tz-aware), dt_ny (tz-aware)
      - date_us_open (python date of NY open)
      - ny_open_ny / ny_close_ny / ny_warmup_end_ny (tz-aware NY)
      - ny_open_utc / ny_close_utc / ny_warmup_end_utc (tz-aware UTC)
    """
    out = df.copy()

    # 1) Ensure UTC ts on every row
    if dt_col not in out.columns:
        raise KeyError(f"Missing timestamp column '{dt_col}'.")
    dt_utc = pd.to_datetime(out[dt_col], utc=True, errors="coerce")
    if dt_utc.isna().any():
        raise ValueError("Unparsable datetimes found.")
    out["dt_utc"] = dt_utc

    # 2) Convert to NY local
    z = ZoneInfo(tz_market)
    out["dt_ny"] = out["dt_utc"].dt.tz_convert(z)

    # 3) Compute the NY-local open/close for each ROW's NY date (DST-safe)
    ny_date_str = out["dt_ny"].dt.strftime("%Y-%m-%d")
    ny_open_ny = pd.to_datetime(ny_date_str + f" {market_open_local}")\
                    .dt.tz_localize(z, ambiguous="infer", nonexistent="shift_forward")
    ny_close_ny = pd.to_datetime(ny_date_str + f" {market_close_local}")\
                    .dt.tz_localize(z, ambiguous="infer", nonexistent="shift_forward")
    ny_warmup_end_ny = ny_open_ny + pd.to_timedelta(warmup_minutes, unit="m")

    out["ny_open_ny"] = ny_open_ny
    out["ny_close_ny"] = ny_close_ny
    out["ny_warmup_end_ny"] = ny_warmup_end_ny

    # 4) UTC versions of those markers (for unambiguous comparisons/exports)
    out["ny_open_utc"] = out["ny_open_ny"].dt.tz_convert("UTC")
    out["ny_close_utc"] = out["ny_close_ny"].dt.tz_convert("UTC")
    out["ny_warmup_end_utc"] = out["ny_warmup_end_ny"].dt.tz_convert("UTC")

    # 5) NY trading day key = calendar date of the open in NY
    out["date_us_open"] = out["ny_open_ny"].dt.date

    return out
