from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.feature_time import add_ny_session_markers
except Exception:
    try:
        from .feature_time import add_ny_session_markers
    except Exception:
        from feature_time import add_ny_session_markers


def _ensure_session_markers(df: pd.DataFrame,
                            *,
                            dt_col: str = "Gmt time",
                            warmup_minutes: int = 30) -> pd.DataFrame:
    x = df.copy()
    if "dt_utc" not in x.columns:
        x["dt_utc"] = pd.to_datetime(x[dt_col], utc=True)
    if "ny_warmup_end_utc" not in x.columns:
        if "ny_open_utc" in x.columns:
            x["ny_warmup_end_utc"] = x["ny_open_utc"] + pd.Timedelta(minutes=warmup_minutes)
    return x


def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3.0


def _vwap_24h_up_to(df: pd.DataFrame, cutoff_utc: pd.Timestamp) -> float | np.nan:
    start = cutoff_utc - pd.Timedelta(hours=24)
    win = df[(df["dt_utc"] > start) & (df["dt_utc"] <= cutoff_utc)]
    if win.empty or (win["Volume"].fillna(0) <= 0).all():
        return np.nan
    tp = _typical_price(win)
    return float((tp * win["Volume"]).sum() / win["Volume"].sum())


def compute_levels(
    df,
    *,
    ts_col: str = "Gmt time",
    fv_window_minutes: int = 30,
    vwap_mode: str = "rolling_24h",
    vwap_alpha: float = 0.25,
    l1_use: bool = True,
    l1_mode: str = "ib_multiple",
    ib_k: float = 1.0,
    vol_score_col: str = "vol_score",
    vol_scale_l1: bool = True,
    vol_scale_fv: bool = False,
    # --- NEW ---
    scale_mode: str = "up_only",     # "up_only" | "both" | "none"
    cap_gap_lo: float | None = None,
    cap_gap_hi: float | None = None,
) -> pd.DataFrame:

    """
    Compute FV and L1 levels, carry prev day until warmup end.
    """
    x = _ensure_session_markers(df, dt_col=ts_col).sort_values("dt_utc").reset_index(drop=True)

    per_day = []
    for day, g in x.groupby("date_us_open", sort=True):
        ib_start = g["ny_open_utc"].iloc[0]
        ib_end   = g["ny_warmup_end_utc"].iloc[0]

        ib = g[(g["dt_utc"] >= ib_start) & (g["dt_utc"] < ib_end)]
        if ib.empty:
            continue

        fv_high = float(ib["High"].max())
        fv_low  = float(ib["Low"].min())
        fv_mid  = 0.5 * (fv_high + fv_low)
        half_range = 0.5 * (fv_high - fv_low)

        # optional vol scaling of FV width
        fv_width_scale = 1.0
        if vol_scale_fv and vol_score_col in g.columns:
            vscore = g[vol_score_col].iloc[0]
            if pd.notna(vscore):
                fv_width_scale = float(vscore)

        vwap = _vwap_24h_up_to(g, cutoff_utc=ib_end) if vwap_mode=="rolling_24h" else np.nan
        fv_mid_adj = fv_mid if np.isnan(vwap) else (1-vwap_alpha)*fv_mid + vwap_alpha*vwap

        half_range_adj = half_range * fv_width_scale
        fv_low_adj, fv_high_adj = fv_mid_adj-half_range_adj, fv_mid_adj+half_range_adj
        fv_half_dn, fv_half_up = 0.5*(fv_mid_adj+fv_low_adj), 0.5*(fv_mid_adj+fv_high_adj)

                # Base distance from FV → L1
        gap = (fv_high - fv_low) * ib_k

        # Volatility scaling for L1 gap
        if l1_use and vol_scale_l1 and (vol_score_col in g.columns):
            vs = g[vol_score_col].iloc[0]
            if pd.notna(vs):
                if scale_mode == "up_only":
                    gap *= max(1.0, float(vs))   # expand on high vol, never shrink
                elif scale_mode == "both":
                    gap *= float(vs)             # expand or shrink
                elif scale_mode == "none":
                    pass

        # Optional hard caps if you still use them
        if cap_gap_lo is not None:
            gap = max(gap, cap_gap_lo)
        if cap_gap_hi is not None:
            gap = min(gap, cap_gap_hi)


        if l1_use:
            l1_up = fv_high_adj+gap; l1_dn=fv_low_adj-gap
            l1_mid_up=0.5*(fv_high_adj+l1_up); l1_mid_dn=0.5*(fv_low_adj+l1_dn)
        else:
            l1_up=l1_dn=l1_mid_up=l1_mid_dn=np.nan

        per_day.append(dict(
            date_us_open=day,
            FV_low_adj=fv_low_adj, FV_mid_adj=fv_mid_adj, FV_high_adj=fv_high_adj,
            FV_half_dn=fv_half_dn, FV_half_up=fv_half_up,
            L1_dn=l1_dn, L1_mid_dn=l1_mid_dn, L1_mid_up=l1_mid_up, L1_up=l1_up,
        ))

    if not per_day:
        return x

    day_df = pd.DataFrame(per_day).set_index("date_us_open").sort_index()
    day_prev = day_df.shift(1).add_suffix("_prev")
    day_all  = pd.concat([day_df, day_prev], axis=1)

    out = x.merge(day_all, left_on="date_us_open", right_index=True, how="left")

    after_warmup = out["dt_utc"] >= out["ny_warmup_end_utc"]
    for c in ["FV_low_adj","FV_mid_adj","FV_high_adj","FV_half_dn","FV_half_up",
              "L1_dn","L1_mid_dn","L1_mid_up","L1_up"]:
        out[f"{c}_active"] = np.where(after_warmup, out[c], out[f"{c}_prev"])

    return out
