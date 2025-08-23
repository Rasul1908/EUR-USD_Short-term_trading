# --- src/feature_volatility.py ---
from __future__ import annotations

import numpy as np
import pandas as pd

# Resilient import: package, relative, then local (script mode)
try:
    from src.feature_time import add_ny_session_markers  # imported as package
except Exception:  # noqa: BLE001
    try:
        from .feature_time import add_ny_session_markers  # imported relatively
    except Exception:  # noqa: BLE001
        from feature_time import add_ny_session_markers   # run as bare script


# ---------- safety helpers ----------
def _to_utc(ts: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(ts):
        if ts.dt.tz is None:
            return ts.dt.tz_localize("UTC")
        return ts.dt.tz_convert("UTC")
    return pd.to_datetime(ts, utc=True)


def _ensure_session_markers(df: pd.DataFrame,
                            *,
                            dt_col: str = "Gmt time",
                            warmup_minutes: int = 30) -> pd.DataFrame:
    """
    Ensure we have: dt_utc, ny_open_utc, ny_close_utc, ny_warmup_end_utc, date_us_open.
    Create from NY-local markers if needed. Compute warmup_end if missing.
    """
    x = df.copy()

    if "dt_utc" not in x.columns:
        x["dt_utc"] = pd.to_datetime(x[dt_col], utc=True)

    needs_time = not ({"ny_open_utc", "ny_close_utc", "ny_warmup_end_utc", "date_us_open"} <= set(x.columns) or
                      {"ny_open_ny", "ny_close_ny", "date_us_open"} <= set(x.columns))
    if needs_time:
        x = add_ny_session_markers(x, dt_col=dt_col)

    if "ny_open_utc" not in x.columns and "ny_open_ny" in x.columns:
        x["ny_open_utc"] = _to_utc(x["ny_open_ny"])
    if "ny_close_utc" not in x.columns and "ny_close_ny" in x.columns:
        x["ny_close_utc"] = _to_utc(x["ny_close_ny"])
    if "ny_warmup_end_utc" not in x.columns:
        if "ny_warmup_end_ny" in x.columns:
            x["ny_warmup_end_utc"] = _to_utc(x["ny_warmup_end_ny"])
        elif "ny_open_utc" in x.columns:
            x["ny_warmup_end_utc"] = x["ny_open_utc"] + pd.Timedelta(minutes=warmup_minutes)

    missing = [c for c in ["ny_open_utc", "ny_close_utc", "ny_warmup_end_utc", "date_us_open"]
               if c not in x.columns]
    if missing:
        raise KeyError(f"Missing session markers after synthesis: {missing}")

    return x


# ---------- core aggregates ----------
def _preus_window(g: pd.DataFrame) -> pd.DataFrame:
    """Return the pre-US slice for a given date_us_open group."""
    return g[g["dt_utc"] < g["ny_open_utc"]]


def _preus_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per US day (date_us_open), compute pre-US high/low/range.
    Returns one row per date_us_open.
    """
    recs = []
    for day, g in df.groupby("date_us_open", sort=True):
        pre = _preus_window(g)
        if pre.empty:
            continue
        recs.append(dict(
            date_us_open=day,
            preus_high=float(pre["High"].max()),
            preus_low=float(pre["Low"].min()),
            preus_range=float(pre["High"].max() - pre["Low"].min()),
        ))
    return pd.DataFrame.from_records(recs)


# ---------- public API ----------
def attach_volatility_score(
    df: pd.DataFrame,
    *,
    dt_col: str = "Gmt time",
    warmup_minutes: int = 30,
    atr_lookback: int = 14,         # <-- ATR window (days)
    atr_method: str = "sma",        # "sma" | "ema"
    k_atr: float = 1.20,            # <-- volatility threshold: vol if range >= k_atr * ATR
    cap_lo: float = 0.7,            # caps for vol_score (used downstream to scale L1)
    cap_hi: float = 1.3,
    add_binary_flag: bool = True,   # keep binary 'is_volatile' for convenience
) -> pd.DataFrame:
    """
    Compute ATR-relative volatility regime.

    Steps:
      1) preUS daily range per date_us_open
      2) ATR of preUS range over `atr_lookback` days (SMA or EMA)
      3) vol_score_raw = preUS_range / ATR_preUS
      4) vol_score = clip(vol_score_raw, [cap_lo, cap_hi])
      5) is_volatile = (vol_score_raw >= k_atr)

    Returns the original dataframe with 'vol_score' and (optionally) 'is_volatile' broadcast to rows.
    """
    x = _ensure_session_markers(df, dt_col=dt_col, warmup_minutes=warmup_minutes)
    x = x.sort_values("dt_utc").reset_index(drop=True)

    daily = _preus_aggregates(x)
    if daily.empty:
        out = x.copy()
        out["vol_score"] = np.nan
        if add_binary_flag:
            out["is_volatile"] = pd.NA
        return out

    daily = daily.sort_values("date_us_open").reset_index(drop=True)

    # --- ATR of preUS range ---
    if atr_method.lower() == "ema":
        alpha = 2.0 / (atr_lookback + 1.0)
        daily["atr_preus"] = daily["preus_range"].ewm(alpha=alpha, adjust=False, min_periods=3).mean()
    else:  # SMA
        daily["atr_preus"] = daily["preus_range"].rolling(atr_lookback, min_periods=3).mean()

    # avoid divide-by-zero
    eps = 1e-12
    daily["vol_score_raw"] = daily["preus_range"] / (daily["atr_preus"] + eps)
    daily["vol_score"] = daily["vol_score_raw"].clip(lower=cap_lo, upper=cap_hi)

    if add_binary_flag:
        daily["is_volatile"] = (daily["vol_score_raw"] >= k_atr).astype("Int64")

    # broadcast back to intraday
    cols = ["date_us_open", "vol_score"] + (["is_volatile"] if add_binary_flag else [])
    out = x.merge(daily[cols], on="date_us_open", how="left")

    return out
