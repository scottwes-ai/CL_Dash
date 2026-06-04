"""
CL Signal Dashboard -- SINGLE FILE, no external signal modules required.
Deploy with only cl_module.py + requirements.txt.

Strategy: NYMEX WTI Crude Oil Roll Yield  |  MA13+stor (full)

  Signal    : (CL1−CL2)/CL1×100 spread sign each week
              Spread ≥ 0 → backwardation → LONG
              Spread  < 0 → contango     → SHORT
  Filter    : Short requires BOTH conditions:
                (1) CL1 price < 13-week MA of CL1  [bearish trend]
                (2) storage > 26-week MA of storage [oversupply]
              Longs are unfiltered.
  Entry     : C2 price on signal change
  Roll exit : C1 price (captures curve convergence)
  Signal exit: C2 price on spread sign flip
  Roll date : last Friday before NYMEX expiry
              (3 bdays before 25th of prior month; 4 if 25th is weekend)
"""

import io
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ==============================================================================
# EIA DATA FETCHING
# ==============================================================================

EIA_CL_PRICE_XLS_URL = "https://www.eia.gov/dnav/pet/xls/PET_PRI_FUT_S1_W.xls"
EIA_CL_STORAGE_API   = (
    "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    "?api_key=DEMO_KEY&frequency=weekly&data[0]=value"
    "&facets[product][]=EPC0&facets[duoarea][]=NUS&facets[process][]=SAX"
    "&sort[0][column]=period&sort[0][direction]=asc&length=5000&offset=0"
)
_HDRS = {"User-Agent": "Mozilla/5.0 (compatible; CL-Dashboard/1.0)"}


def load_cl_prices() -> pd.DataFrame:
    """
    Load weekly WTI crude oil futures (C1/C2) from EIA Excel file.
    Falls back to yfinance if EIA XLS is unavailable.
    Returns DataFrame: date (Fridays), CL1, CL2
    """
    try:
        r = requests.get(EIA_CL_PRICE_XLS_URL, headers=_HDRS, timeout=60)
        r.raise_for_status()
        xl = pd.ExcelFile(io.BytesIO(r.content), engine="xlrd")
        df = None
        for sheet in xl.sheet_names:
            try:
                raw = xl.parse(sheet, header=None, skiprows=3)
                if raw.shape[1] < 3: continue
                raw.columns = list(range(raw.shape[1]))
                raw[0] = pd.to_datetime(raw[0], errors="coerce")
                raw = raw.dropna(subset=[0])
                if (raw[0].dt.year.between(1980, 2030)).sum() > 100:
                    df = raw; break
            except Exception:
                continue
        if df is None:
            raise RuntimeError("Could not parse EIA crude XLS.")
        df = df.rename(columns={0: "date", 1: "CL1", 2: "CL2"})
        for c in ["CL1", "CL2"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = (df.dropna(subset=["date", "CL1", "CL2"])
                .sort_values("date").reset_index(drop=True))
        df = df[df["date"].dt.dayofweek == 4].reset_index(drop=True)
        return df[["date", "CL1", "CL2"]]
    except Exception:
        return _load_cl_prices_yfinance()


def _load_cl_prices_yfinance(lookback_years: int = 40) -> pd.DataFrame:
    """Fallback: build weekly C1/C2 from individual CL contracts via yfinance."""
    import yfinance as yf
    today = date.today()
    ty, tm = today.year, today.month
    sy, sm = _add_months(ty, tm, -lookback_years * 12)
    ey, em = _add_months(ty, tm, 4)
    contracts = []
    cy, cm = max(sy, 1990), sm if sy >= 1990 else 1
    while (cy, cm) <= (ey, em):
        contracts.append(_cl_ticker(cy, cm)); cy, cm = _add_months(cy, cm, 1)

    start_str = f"{max(sy, 1990)}-01-01"
    daily = {}
    for sym in contracts:
        try:
            h = yf.Ticker(sym).history(start=start_str, interval="1d")
            if not h.empty:
                h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
                daily[sym] = h["Close"].rename(sym)
        except Exception:
            pass

    if not daily:
        raise RuntimeError("yfinance returned no CL data.")

    df_d = pd.DataFrame(daily).sort_index()
    df_d = df_d[~df_d.index.duplicated(keep="last")]
    sorted_syms = sorted(df_d.columns, key=_contract_sort_key)
    df_d = df_d[sorted_syms]
    df_fri = df_d[df_d.index.dayofweek == 4].copy()
    rows = []
    for dt in df_fri.index:
        if dt.date() > today: continue
        rp = df_fri.loc[dt].dropna()
        vs = [s for s in sorted_syms if s in rp.index]
        if len(vs) < 2: continue
        rows.append({"date": dt, "CL1": float(rp[vs[0]]), "CL2": float(rp[vs[1]])})
    if not rows:
        raise RuntimeError("No complete C1/C2 rows from yfinance.")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_cl_storage() -> pd.DataFrame:
    """EIA U.S. Crude Oil Ending Stocks excl. SPR via API v2."""
    r = requests.get(EIA_CL_STORAGE_API, headers=_HDRS, timeout=60)
    r.raise_for_status()
    data = r.json()
    records = data.get("response", {}).get("data", [])
    if not records:
        raise RuntimeError("EIA storage API returned no data.")
    rows = [{"date": pd.to_datetime(rec["period"]),
             "storage_kbbl": float(rec["value"])}
            for rec in records if rec.get("value")]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return _compute_storage_features(df)


def _compute_storage_features(stor: pd.DataFrame) -> pd.DataFrame:
    stor = stor.copy().sort_values("date").reset_index(drop=True)
    stor["week_of_year"] = stor["date"].dt.isocalendar().week.astype(int)
    stor["stor_ma26"]    = stor["storage_kbbl"].rolling(26).mean()
    stor["stor_above_ma"] = (stor["storage_kbbl"] > stor["stor_ma26"]).astype(float)
    stor["stor_chg13"]   = stor["storage_kbbl"].diff(13)
    return stor


# ==============================================================================
# CONTRACT / ROLL DATE UTILITIES
# ==============================================================================

_CL_MONTH_CODES = "FGHJKMNQUVXZ"


def _add_months(y: int, m: int, n: int) -> tuple:
    m += n
    while m > 12: m -= 12; y += 1
    while m < 1:  m += 12; y -= 1
    return y, m


def _cl_ticker(year: int, month: int) -> str:
    return f"CL{_CL_MONTH_CODES[month - 1]}{str(year)[-2:]}.NYM"


def _contract_sort_key(sym: str) -> tuple:
    code = sym[2]; yr2 = int(sym[3:5])
    year = 2000 + yr2 if yr2 < 50 else 1900 + yr2
    return (year, _CL_MONTH_CODES.index(code) + 1)


def cl_expiry(contract_month: int, contract_year: int) -> date:
    """
    NYMEX CL: trading terminates 3 business days before 25th of prior month.
    4 business days if the 25th is a Saturday or Sunday.
    """
    exp_month = contract_month - 1 if contract_month > 1 else 12
    exp_year  = contract_year  if contract_month > 1 else contract_year - 1
    d25 = date(exp_year, exp_month, 25)
    bdays = 4 if d25.weekday() >= 5 else 3
    cur = d25 - timedelta(1); count = 0
    while count < bdays:
        if cur.weekday() < 5: count += 1
        if count < bdays: cur -= timedelta(1)
    return cur


def last_fri_before(d: date) -> date:
    cur = d - timedelta(1)
    while cur.weekday() != 4: cur -= timedelta(1)
    return cur


def _bar_close_friday(d: date) -> date:
    """Friday of the bar week (TradingView bars open on Monday)."""
    return d + timedelta((4 - d.weekday()) % 7)


def build_roll_set(yr_min: int, yr_max: int) -> set:
    s = set()
    for yr in range(yr_min - 1, yr_max + 2):
        for mo in range(1, 13):
            s.add(last_fri_before(cl_expiry(mo, yr)))
    return s


def next_roll_date(today: date) -> date:
    for yr in range(today.year, today.year + 2):
        for mo in range(1, 13):
            rd = last_fri_before(cl_expiry(mo, yr))
            if rd >= today: return rd
    return None


# ==============================================================================
# SIGNAL ENGINE
# ==============================================================================

def _prepare_weekly_df(prices: pd.DataFrame, stor: pd.DataFrame) -> pd.DataFrame:
    """
    Merge prices + storage; compute spread, price MA13, storage MA26,
    roll flags, and filtered signal.

    Signal:  spread = (CL1-CL2)/CL1*100
             LONG  if spread >= 0  (backwardation)
             SHORT if spread  < 0  (contango)
    Filter:  shorts require CL1 < price_ma13  AND  storage > stor_ma26
    """
    df = prices[["date", "CL1", "CL2"]].copy().sort_values("date").reset_index(drop=True)

    # Date helpers for bar→Friday matching
    df["date_d"]  = df["date"].dt.date
    df["bar_fri"] = df["date_d"].apply(_bar_close_friday)

    # Roll weeks
    roll_set = build_roll_set(df["date"].min().year, df["date"].max().year)
    df["is_roll"] = df["bar_fri"].apply(
        lambda f: (f in roll_set) or ((f - timedelta(1)) in roll_set))

    # Spread and filter indicators
    df["spread"]      = (df["CL1"] - df["CL2"]) / df["CL1"] * 100
    df["price_ma13"]  = df["CL1"].rolling(13, min_periods=13).mean()

    # Merge storage — both bar_fri and EIA dates are Fridays
    stor_copy = stor[["date", "storage_kbbl", "stor_ma26", "stor_above_ma",
                       "stor_chg13"]].copy()
    stor_copy["bar_fri"] = stor_copy["date"].dt.date
    df = df.merge(stor_copy.drop(columns=["date"]), on="bar_fri", how="left")
    for col in ["storage_kbbl", "stor_ma26", "stor_above_ma", "stor_chg13"]:
        df[col] = df[col].ffill()

    # Raw signal: spread sign, no MA
    df["raw_signal"] = np.where(df["spread"] >= 0, "LONG", "SHORT")

    # Filter flags
    df["price_below_ma13"] = (df["CL1"] < df["price_ma13"]).astype(float)
    df["stor_above_ma26"]  = (df["storage_kbbl"] > df["stor_ma26"]).astype(float)
    df["filter_short_ok"]  = (df["price_below_ma13"] == 1) & (df["stor_above_ma26"] == 1)

    # Filtered signal: shorts blocked unless both filter conditions pass
    blocked = (df["raw_signal"] == "SHORT") & (~df["filter_short_ok"])
    # Also block while price_ma13 hasn't warmed up (first 13 bars)
    no_ma = df["price_ma13"].isna()
    df["filtered_signal"] = np.where(no_ma, "WARMUP",
                             np.where(blocked, "FLAT", df["raw_signal"]))
    return df.reset_index(drop=True)


def _run_backtest_m2m(df: pd.DataFrame, sig_col: str) -> pd.Series:
    """
    Weekly mark-to-market returns.
    Hold previous week's position; exit at C1 (roll week) or C2 (non-roll).
    Returns are NaN for warmup/flat weeks.
    """
    pos_map = {"LONG": 1.0, "SHORT": -1.0, "FLAT": 0.0, "WARMUP": 0.0}
    pos  = df[sig_col].map(pos_map).fillna(0.0).values
    c1   = df["CL1"].values
    c2   = df["CL2"].values
    roll = df["is_roll"].values
    ret  = np.full(len(df), 0.0)
    for i in range(1, len(df)):
        pp = pos[i - 1]                    # last week's position
        if pp == 0.0: continue
        exit_p = c1[i] if roll[i] else c2[i]
        if c2[i - 1] > 0:
            ret[i] = (exit_p / c2[i - 1] - 1.0) * pp
    return pd.Series(ret, index=pd.to_datetime(df["date"].values))


def _run_cl_backtest_trades(df: pd.DataFrame, filtered: bool = True) -> pd.DataFrame:
    """
    Trade-by-trade log.  Each trade spans one roll period (or to signal flip).
    Entry: C2.  Roll exit: C1.  Signal/filter exit: C2.
    """
    sig_col = "filtered_signal" if filtered else "raw_signal"
    pos_map = {"LONG": 1, "SHORT": -1, "FLAT": 0, "WARMUP": 0}
    rows_df = df[df["price_ma13"].notna()].reset_index(drop=True)
    trades  = []; active_dir = 0; entry_row = None

    for _, row in rows_df.iterrows():
        raw_dir   = 1 if row["spread"] >= 0 else -1
        is_roll   = bool(row["is_roll"])

        if filtered:
            if raw_dir == -1:
                direction_want = -1 if row["filter_short_ok"] else 0
            else:
                direction_want = 1
        else:
            direction_want = raw_dir

        # ── Roll exit ─────────────────────────────────────────────────────
        if is_roll and active_dir != 0:
            ep  = float(entry_row["CL2"])
            xp  = float(row["CL1"])
            ret = (xp / ep - 1.0) * active_dir
            trades.append(_make_trade(entry_row, row, active_dir, ep, xp, ret, "roll"))
            active_dir = 0; entry_row = None

        # ── Signal / filter flip exit ─────────────────────────────────────
        elif not is_roll and active_dir != 0 and direction_want != active_dir:
            ep  = float(entry_row["CL2"])
            xp  = float(row["CL2"])
            ret = (xp / ep - 1.0) * active_dir
            exit_type = "signal" if raw_dir != active_dir else "filter"
            trades.append(_make_trade(entry_row, row, active_dir, ep, xp, ret, exit_type))
            active_dir = 0; entry_row = None

        # ── Enter ─────────────────────────────────────────────────────────
        if active_dir == 0 and direction_want != 0:
            active_dir = direction_want; entry_row = row

    return pd.DataFrame(trades)


def _make_trade(er, xr, direction: int, ep: float, xp: float,
                ret: float, exit_type: str) -> dict:
    def _s(row, k):
        v = row.get(k, np.nan); return float(v) if pd.notna(v) else None
    return {
        "entry_date":       er["date"],
        "exit_date":        xr["date"],
        "signal":           "LONG" if direction == 1 else "SHORT",
        "exit_type":        exit_type,
        "entry_price":      round(ep, 2),
        "exit_price":       round(xp, 2),
        "return":           round(ret, 6),
        "hold_weeks":       max(1, round((xr["date"] - er["date"]).days / 7)),
        "spread_entry":     round(float(er["spread"]), 4),
        "CL1_entry":        round(float(er["CL1"]), 2),
        "price_ma13_entry": round(_s(er, "price_ma13") or 0.0, 2),
        "storage_entry":    round(_s(er, "storage_kbbl") or 0.0, 0),
        "stor_ma26_entry":  round(_s(er, "stor_ma26") or 0.0, 0),
        "is_roll_exit":     exit_type == "roll",
    }


def _compute_metrics_m2m(ret_series: pd.Series, total_yrs: float) -> dict:
    r   = ret_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 10: return {}
    eq  = (1 + r).cumprod(); feq = float(eq.iloc[-1])
    cagr = feq ** (1 / total_yrs) - 1 if total_yrs > 0 else np.nan
    ann_r = r.mean() * 52; ann_v = r.std() * np.sqrt(52)
    sharpe = ann_r / ann_v if ann_v > 0 else np.nan
    down   = r[r < 0]; dd_std = float(down.std(ddof=1)) if len(down) > 1 else 1e-9
    sortino = ann_r / (dd_std * np.sqrt(52)) if dd_std > 0 else np.nan
    mdd    = float(((eq / eq.cummax()) - 1).min())
    wins   = r[r > 0]; loses = r[r < 0]
    return {
        "cagr_pct":      round(cagr * 100, 2),
        "sharpe":        round(float(sharpe), 2),
        "sortino":       round(float(sortino), 2),
        "max_dd_pct":    round(mdd * 100, 2),
        "win_rate_pct":  round((r > 0).mean() * 100, 1),
        "avg_win_pct":   round(wins.mean() * 100, 3) if len(wins) > 0 else 0.0,
        "avg_loss_pct":  round(loses.mean() * 100, 3) if len(loses) > 0 else 0.0,
        "skew":          round(float(r.skew()), 2) if len(r) > 2 else None,
        "weeks":         len(r),
        "final_equity":  round(feq, 4),
    }


# ==============================================================================
# LIVE SIGNAL (yfinance)
# ==============================================================================

def _fetch_live_cl(today: date, lookback_weeks: int = 20) -> pd.DataFrame:
    import yfinance as yf
    ty, tm = today.year, today.month
    sy, sm = _add_months(ty, tm, -6)
    ey, em = _add_months(ty, tm, 4)
    contracts = []
    cy, cm = sy, sm
    while (cy, cm) <= (ey, em):
        contracts.append(_cl_ticker(cy, cm)); cy, cm = _add_months(cy, cm, 1)

    start_str = (today - timedelta(weeks=lookback_weeks + 8)).strftime("%Y-%m-%d")
    daily = {}
    for sym in contracts:
        try:
            h = yf.Ticker(sym).history(start=start_str, interval="1d")
            if not h.empty and len(h) >= 2:
                h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
                daily[sym] = h["Close"].rename(sym)
        except Exception:
            pass

    if not daily:
        raise RuntimeError("yfinance returned no CL data.")

    df_d = pd.DataFrame(daily).sort_index()
    df_d = df_d[~df_d.index.duplicated(keep="last")]
    sorted_syms = sorted(df_d.columns, key=_contract_sort_key)
    df_d = df_d[sorted_syms]
    df_fri = df_d[df_d.index.dayofweek == 4].copy()

    rows = []
    for dt in df_fri.index:
        if dt.date() > today: continue
        rp = df_fri.loc[dt].dropna()
        vs = [s for s in sorted_syms if s in rp.index]
        if len(vs) < 2: continue
        rows.append({"date": dt, "CL1": float(rp[vs[0]]), "CL2": float(rp[vs[1]]),
                     "C1_sym": vs[0], "C2_sym": vs[1]})
    if not rows:
        raise RuntimeError("No complete C1/C2 rows from yfinance.")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["spread"]     = (df["CL1"] - df["CL2"]) / df["CL1"] * 100
    df["price_ma13"] = df["CL1"].rolling(13, min_periods=13).mean()
    return df


def _compute_live_signal(stor_df: pd.DataFrame) -> dict:
    today    = date.today()
    df_live  = _fetch_live_cl(today)
    latest   = df_live.dropna(subset=["price_ma13"]).iloc[-1]
    cl1_val  = float(latest["CL1"])
    cl2_val  = float(latest["CL2"])
    pm13_val = float(latest["price_ma13"])
    spread   = float(latest["spread"])
    raw_sig  = "LONG" if spread >= 0 else "SHORT"

    # Storage filter
    stor_sub = stor_df[stor_df["date"] <= pd.Timestamp(today)].dropna(subset=["stor_ma26"])
    if stor_sub.empty:
        storage_kbbl = np.nan; stor_ma26 = np.nan; stor_date = None
        filter_price = True;   filter_stor  = True
        filter_reason = "No storage data — filter inactive"
    else:
        sl = stor_sub.iloc[-1]
        storage_kbbl = float(sl["storage_kbbl"]); stor_ma26 = float(sl["stor_ma26"])
        stor_date    = sl["date"].date()
        filter_price = cl1_val < pm13_val
        filter_stor  = storage_kbbl > stor_ma26

    if raw_sig == "SHORT":
        filter_pass = filter_price and filter_stor
        parts = []
        if not filter_price:
            parts.append(f"CL1 ${cl1_val:.2f} NOT below MA13 ${pm13_val:.2f}")
        if not filter_stor:
            parts.append(f"Storage {storage_kbbl:,.0f} kBbl NOT above MA26 {stor_ma26:,.0f}")
        if filter_pass:
            filter_reason = (f"SHORT allowed — CL1 ${cl1_val:.2f} < MA13 ${pm13_val:.2f}"
                             f"  AND  storage {storage_kbbl:,.0f} > MA26 {stor_ma26:,.0f} kBbl")
        else:
            filter_reason = "SHORT blocked — " + "; ".join(parts)
    else:
        filter_pass   = True
        filter_reason = f"LONG — no filter applied (spread = {spread:+.4f}%)"

    final_signal = raw_sig if filter_pass else "FLAT"
    nr = next_roll_date(today)

    # Build spread history for audit (last 8 rows)
    spread_tbl = df_live.tail(8)[["date", "C1_sym", "CL1", "C2_sym", "CL2",
                                   "spread", "price_ma13"]].copy()
    spread_tbl["date"] = pd.to_datetime(spread_tbl["date"])

    return {
        "date":           today,
        "CL1":            cl1_val,
        "CL2":            cl2_val,
        "C1_sym":         str(latest.get("C1_sym", "")),
        "C2_sym":         str(latest.get("C2_sym", "")),
        "spread":         spread,
        "price_ma13":     pm13_val,
        "raw_signal":     raw_sig,
        "filter_price":   filter_price,
        "filter_stor":    filter_stor,
        "filter_pass":    filter_pass,
        "filter_reason":  filter_reason,
        "final_signal":   final_signal,
        "storage_kbbl":   storage_kbbl,
        "stor_ma26":      stor_ma26,
        "stor_date":      stor_date,
        "next_roll_date": nr,
        "days_to_roll":   (nr - today).days if nr else None,
        "is_roll_week":   (last_fri_before(cl_expiry(today.month, today.year)) == today),
        "spread_table":   spread_tbl,
    }


# ==============================================================================
# COLOURS & HELPERS
# ==============================================================================

C_LONG  = "#00c853"; C_SHORT = "#ff1744"; C_FLAT  = "#9e9e9e"
C_BLUE  = "#1565c0"; C_ORANGE= "#ff6d00"; C_TEAL  = "#00bcd4"
C_BG    = "#0e1117"; C_PANEL = "#1c1e26"; C_MUTED = "#8b8fa8"; C_GRID = "#1e2030"
C_AMBER = "#ffb300"

def _sc(s): return {"LONG": C_LONG, "SHORT": C_SHORT}.get(s, C_FLAT)
def _se(s): return {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⚪"}.get(s, "⚪")


# ==============================================================================
# CACHED LOADERS
# ==============================================================================

@st.cache_data(ttl=43200, show_spinner=False)
def _load_historical():
    prices = load_cl_prices()
    stor   = load_cl_storage()
    df     = _prepare_weekly_df(prices, stor)
    rf     = _run_backtest_m2m(df, "filtered_signal")
    rb     = _run_backtest_m2m(df, "raw_signal")
    tf     = _run_cl_backtest_trades(df, filtered=True)
    tb     = _run_cl_backtest_trades(df, filtered=False)
    return prices, stor, df, rf, rb, tf, tb


@st.cache_data(ttl=300, show_spinner=False)
def _load_live():
    stor = load_cl_storage()
    sig  = _compute_live_signal(stor)
    return sig, stor


# ==============================================================================
# RENDER — HERO CARD
# ==============================================================================

def _hero(sig: dict):
    final = sig["final_signal"]; color = _sc(final); emoji = _se(final)
    bc    = f"background:{color}22;border:2px solid {color};border-radius:12px;"

    if   final == "LONG":  action = "BUY C2";        contract = sig["C2_sym"]; price = sig["CL2"]
    elif final == "SHORT": action = "SELL SHORT C2";  contract = sig["C2_sym"]; price = sig["CL2"]
    else:                  action = "HOLD FLAT";      contract = "— no position"; price = None

    nr   = sig.get("next_roll_date"); dr = sig.get("days_to_roll", 0)
    sprd = sig["spread"]; sc = C_LONG if sprd >= 0 else C_SHORT
    pm13 = sig["price_ma13"]
    fp_c = C_LONG  if sig["filter_price"] else C_SHORT
    fs_c = C_LONG  if sig["filter_stor"]  else C_SHORT
    fp_l = "✓" if sig["filter_price"] else "✗"
    fs_l = "✓" if sig["filter_stor"]  else "✗"
    nrs  = nr.strftime("%b %d") if nr else "—"
    drs  = f"({dr}d)" if dr else ""
    prs  = f"@ ${price:.2f}/bbl" if price else ""

    st.html(f"""
    <div style="{bc} padding:28px 36px;margin-bottom:4px;">
      <div style="display:flex;align-items:flex-start;gap:32px;flex-wrap:wrap;">
        <div style="min-width:200px;">
          <div style="font-size:12px;color:{C_MUTED};letter-spacing:2px;text-transform:uppercase;">
            Signal &middot; {sig['date'].strftime('%A %Y-%m-%d')}</div>
          <div style="font-size:72px;font-weight:900;color:{color};line-height:1;margin-top:4px;">
            {emoji} {final}</div>
          <div style="font-size:13px;color:{C_MUTED};margin-top:6px;">
            Spread Sign  ·  Price &lt; MA13  ·  Stor &gt; MA26</div>
        </div>
        <div style="flex:1;min-width:260px;border-left:1px solid #333;padding-left:28px;">
          <div style="font-size:11px;color:{C_MUTED};text-transform:uppercase;">{action}</div>
          <div style="font-size:24px;font-weight:700;color:white;margin-top:4px;">{contract}</div>
          <div style="font-size:17px;color:#ccc;margin-top:2px;">{prs}</div>
          <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:14px;">
            <div><div style="font-size:11px;color:{C_MUTED};">Spread</div>
              <div style="font-size:19px;font-weight:700;color:{sc};">{sprd:+.4f}%</div></div>
            <div><div style="font-size:11px;color:{C_MUTED};">CL1 vs MA13</div>
              <div style="font-size:19px;font-weight:700;color:{fp_c};">
                {fp_l} ${sig['CL1']:.2f} / ${pm13:.2f}</div></div>
            <div><div style="font-size:11px;color:{C_MUTED};">Short filters</div>
              <div style="font-size:15px;font-weight:700;color:{fp_c};">{fp_l} Price</div>
              <div style="font-size:15px;font-weight:700;color:{fs_c};">{fs_l} Stor</div></div>
            <div><div style="font-size:11px;color:{C_MUTED};">Next roll</div>
              <div style="font-size:19px;font-weight:700;color:#ccc;">
                {nrs} <span style="font-size:12px;color:{C_MUTED};">{drs}</span></div></div>
          </div>
        </div>
        <div style="min-width:155px;text-align:right;">
          <div style="font-size:11px;color:{C_MUTED};margin-bottom:6px;">PRICES (yfinance)</div>
          <div style="font-size:14px;color:#ccc;line-height:2.2;">
            C1 <b style="color:white">${sig['CL1']:.2f}</b>
              <span style="font-size:11px;color:{C_MUTED};">{sig['C1_sym']}</span><br>
            C2 <b style="color:white">${sig['CL2']:.2f}</b>
              <span style="font-size:11px;color:{C_MUTED};">{sig['C2_sym']}</span>
          </div>
        </div>
      </div>
    </div>""")

    if sig["filter_pass"]:
        st.success(f"✅ {sig['filter_reason']}")
    else:
        st.warning(f"🚫 {sig['filter_reason']}")
    if sig.get("is_roll_week"):
        st.error("⚠️ TODAY IS A ROLL DATE — exit position at C1, then re-enter direction at C2.")


# ==============================================================================
# RENDER — SIGNAL AUDIT
# ==============================================================================

def _audit(sig: dict, stor_df: pd.DataFrame):
    c1, c2, c3 = st.columns([1.1, 1.1, 0.9])

    # ── Spread audit ──────────────────────────────────────────────────────────
    with c1:
        st.markdown("#### Spread Audit")
        st.caption("Spread = (CL1−CL2)/CL1×100. LONG if ≥0 (backwardation), SHORT if <0 (contango).")
        tbl = sig["spread_table"].copy()
        tbl["Date"]   = tbl["date"].dt.strftime("%Y-%m-%d")
        tbl.iloc[-1, tbl.columns.get_loc("Date")] += " *"
        disp = tbl[["Date", "C1_sym", "CL1", "CL2", "spread"]].copy()
        disp.columns = ["Date", "C1 Contract", "CL1 $/bbl", "CL2 $/bbl", "Spread%"]
        disp["CL1 $/bbl"] = disp["CL1 $/bbl"].map("${:.2f}".format)
        disp["CL2 $/bbl"] = disp["CL2 $/bbl"].map("${:.2f}".format)
        disp["Spread%"]   = disp["Spread%"].map("{:+.4f}%".format)
        st.dataframe(disp.set_index("Date"), use_container_width=True, height=275)
        sprd = sig["spread"]; sc = C_LONG if sprd >= 0 else C_SHORT
        st.html(f"""
        <div style="background:{C_PANEL};border-radius:8px;padding:12px 16px;">
          <div style="font-size:11px;color:{C_MUTED};text-transform:uppercase;">
            Current spread</div>
          <div style="font-size:24px;font-weight:800;color:{sc};margin-top:4px;">
            {sprd:+.4f}% &nbsp;
            <span style="font-size:13px;color:{C_MUTED};">
              → {"Backwardation / LONG" if sprd >= 0 else "Contango / SHORT"}</span>
          </div>
        </div>""")

    # ── Price + storage filter audit ─────────────────────────────────────────
    with c2:
        st.markdown("#### Short Filter Audit")
        st.caption("Shorts require: CL1 < 13-wk MA(CL1)  AND  storage > 26-wk MA(storage)")

        sr = stor_df.tail(10)[["date", "storage_kbbl", "stor_ma26",
                                "stor_above_ma"]].copy()
        sr = sr[sr["stor_ma26"].notna()]
        sr["date"]         = sr["date"].dt.strftime("%Y-%m-%d")
        sr["storage_kbbl"] = sr["storage_kbbl"].map("{:,.0f}".format)
        sr["stor_ma26"]    = sr["stor_ma26"].map("{:,.0f}".format)
        sr["Abv MA26"]     = sr["stor_above_ma"].map(
            lambda v: "✓ YES" if v == 1 else "✗ no")
        sr = sr.rename(columns={"date": "Week", "storage_kbbl": "kBbl",
                                 "stor_ma26": "MA26 kBbl"}).drop(columns=["stor_above_ma"])
        st.dataframe(sr.set_index("Week"), use_container_width=True, height=275)

        fp_c = C_LONG  if sig["filter_price"] else C_SHORT
        fs_c = C_LONG  if sig["filter_stor"]  else C_SHORT
        fp_v = (f"${sig['CL1']:.2f} < MA13 ${sig['price_ma13']:.2f}" if sig["filter_price"]
                else f"${sig['CL1']:.2f} ≥ MA13 ${sig['price_ma13']:.2f}")
        sk   = sig.get("storage_kbbl"); sm2 = sig.get("stor_ma26")
        sk_s = f"{sk:,.0f}" if sk and not np.isnan(sk) else "—"
        sm_s = f"{sm2:,.0f}" if sm2 and not np.isnan(sm2) else "—"
        fs_v = (f"{sk_s} > MA26 {sm_s} kBbl" if sig["filter_stor"]
                else f"{sk_s} ≤ MA26 {sm_s} kBbl")
        combined_c = C_LONG if (sig["filter_price"] and sig["filter_stor"]) else C_SHORT
        combined_v = "BOTH PASS — Short OK" if (sig["filter_price"] and sig["filter_stor"]) \
                     else "FILTER FAILS — No Short"
        st.html(f"""
        <div style="background:{C_PANEL};border-radius:8px;padding:12px 16px;">
          <div style="font-size:11px;color:{C_MUTED};text-transform:uppercase;">
            Short Filter Check (current week)</div>
          <div style="margin-top:8px;">
            <span style="font-size:13px;color:{fp_c};font-weight:700;">
              {"✓" if sig["filter_price"] else "✗"} Price: {fp_v}</span><br>
            <span style="font-size:13px;color:{fs_c};font-weight:700;
              margin-top:4px;display:block;">
              {"✓" if sig["filter_stor"] else "✗"} Storage: {fs_v}</span>
          </div>
          <div style="font-size:19px;font-weight:900;color:{combined_c};margin-top:8px;">
            {combined_v}</div>
        </div>""")

    # ── Roll calendar ─────────────────────────────────────────────────────────
    with c3:
        st.markdown("#### Roll Calendar")
        st.caption("Last Fri before NYMEX expiry (3 bdays before 25th of prior month).")
        today = sig["date"]; y, m = today.year, today.month; rows = []
        for _ in range(5):
            rd = last_fri_before(cl_expiry(m, y))
            rows.append({
                "Month":     rd.strftime("%b %Y"),
                "Roll Date": rd.strftime("%Y-%m-%d"),
                "Status":    ("Past" if rd < today else
                              "TODAY" if rd == today else f"+{(rd - today).days}d"),
            })
            y, m = _add_months(y, m, 1)
        st.dataframe(pd.DataFrame(rows).set_index("Month"),
                     use_container_width=True, height=215)
        dr = sig.get("days_to_roll", 0) or 0
        nr = sig.get("next_roll_date")
        uc = "#ff1744" if dr <= 5 else ("#ff9800" if dr <= 10 else C_LONG)
        st.html(f"""
        <div style="background:{C_PANEL};border-radius:8px;padding:12px 16px;">
          <div style="font-size:11px;color:{C_MUTED};">Next roll</div>
          <div style="font-size:20px;font-weight:800;color:{uc};">
            {nr.strftime('%b %d, %Y') if nr else '—'}</div>
          <div style="font-size:12px;color:{C_MUTED};">{dr} days away</div>
        </div>""")


# ==============================================================================
# RENDER — CHARTS
# ==============================================================================

def _charts(df: pd.DataFrame, weeks: int = 104):
    recent = df[df["price_ma13"].notna()].tail(weeks).copy()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.27, 0.25, 0.25, 0.23],
        subplot_titles=[
            "Spread % (CL1−CL2)/CL1  [signal driver]",
            "CL1 Price vs 13-week MA  [short filter #1]",
            "Crude Storage vs 26-week MA  [short filter #2]",
            "Final Signal (filtered)",
        ],
        vertical_spacing=0.06,
    )

    # Shaded regime bands
    in_sig = None; t0 = None
    for _, row in recent.iterrows():
        s = row["filtered_signal"]
        if s != in_sig:
            if in_sig and t0:
                fc = ("rgba(0,200,83,0.12)"  if in_sig == "LONG"  else
                      "rgba(255,23,68,0.12)"  if in_sig == "SHORT" else
                      "rgba(158,158,158,0.04)")
                fig.add_vrect(x0=t0, x1=row["date"],
                              fillcolor=fc, layer="below", line_width=0)
            in_sig = s; t0 = row["date"]
    if in_sig and t0 and not recent.empty:
        fc = ("rgba(0,200,83,0.12)"  if in_sig == "LONG"  else
              "rgba(255,23,68,0.12)"  if in_sig == "SHORT" else "rgba(158,158,158,0.04)")
        fig.add_vrect(x0=t0, x1=recent["date"].iloc[-1],
                      fillcolor=fc, layer="below", line_width=0)

    # ── Panel 1: Spread ──────────────────────────────────────────────────────
    bc = [C_LONG if v >= 0 else C_SHORT for v in recent["spread"]]
    fig.add_trace(go.Bar(x=recent["date"], y=recent["spread"], name="Spread %",
                         marker_color=bc, opacity=0.70), row=1, col=1)
    rolls = recent[recent["is_roll"]]
    fig.add_trace(go.Scatter(x=rolls["date"], y=rolls["spread"], mode="markers",
                             name="Roll week",
                             marker=dict(symbol="triangle-up", size=9, color="white")),
                  row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#555", row=1, col=1)

    # ── Panel 2: CL1 price vs MA13 price ─────────────────────────────────────
    fig.add_trace(go.Scatter(x=recent["date"], y=recent["CL1"], name="CL1 $/bbl",
                             line=dict(color=C_ORANGE, width=1.7)), row=2, col=1)
    fig.add_trace(go.Scatter(x=recent["date"], y=recent["price_ma13"], name="MA13(CL1)",
                             line=dict(color=C_TEAL, width=2.0, dash="dash")), row=2, col=1)
    # Fill: green when CL1 < MA13 (short filter passes), red otherwise
    fig.add_trace(go.Scatter(
        x=list(recent["date"]) + list(recent["date"])[::-1],
        y=list(recent["CL1"]) + list(recent["price_ma13"])[::-1],
        fill="toself",
        fillcolor="rgba(0,200,83,0.08)" if (recent["CL1"] < recent["price_ma13"]).mean() > 0.5
                  else "rgba(255,23,68,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    # ── Panel 3: Storage vs MA26 ──────────────────────────────────────────────
    sv = recent["storage_kbbl"].notna()
    if sv.sum() > 0:
        st_mmbbl = recent.loc[sv, "storage_kbbl"] / 1000
        ma_mmbbl = recent.loc[sv, "stor_ma26"] / 1000
        fig.add_trace(go.Scatter(x=recent.loc[sv, "date"], y=st_mmbbl,
                                 name="Storage MMbbl",
                                 line=dict(color=C_BLUE, width=1.7)), row=3, col=1)
        fig.add_trace(go.Scatter(x=recent.loc[sv, "date"], y=ma_mmbbl,
                                 name="MA26 Storage",
                                 line=dict(color=C_AMBER, width=1.8, dash="dash")), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=list(recent.loc[sv, "date"]) + list(recent.loc[sv, "date"])[::-1],
            y=list(st_mmbbl) + list(ma_mmbbl)[::-1],
            fill="toself",
            fillcolor="rgba(255,23,68,0.10)" if st_mmbbl.mean() > ma_mmbbl.mean()
                      else "rgba(0,200,83,0.08)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=3, col=1)

    # ── Panel 4: Signal ───────────────────────────────────────────────────────
    sv2 = [1 if s == "LONG" else (-1 if s == "SHORT" else 0)
           for s in recent["filtered_signal"]]
    sc2 = [C_LONG if s == "LONG" else (C_SHORT if s == "SHORT" else C_FLAT)
           for s in recent["filtered_signal"]]
    fig.add_trace(go.Bar(x=recent["date"], y=sv2, marker_color=sc2,
                         opacity=0.9, name="Signal", showlegend=False), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#555", row=4, col=1)

    fig.update_layout(
        height=720, hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=0),
        plot_bgcolor=C_BG, paper_bgcolor=C_BG,
        font=dict(color="#ccc"), margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.update_xaxes(gridcolor=C_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False)
    fig.update_yaxes(title_text="Spread %",  row=1, col=1)
    fig.update_yaxes(title_text="$/bbl",     row=2, col=1)
    fig.update_yaxes(title_text="MMbbl",     row=3, col=1)
    fig.update_yaxes(tickvals=[-1, 0, 1], ticktext=["SHORT", "FLAT", "LONG"],
                     range=[-1.4, 1.4], title_text="Signal", row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# RENDER — BACKTEST EQUITY
# ==============================================================================

def _equity_chart(df: pd.DataFrame, rf: pd.Series, rb: pd.Series, total_yrs: float):
    mf = _compute_metrics_m2m(rf, total_yrs)
    mb = _compute_metrics_m2m(rb, total_yrs)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Filtered CAGR",  f"{mf.get('cagr_pct', 0):+.1f}%")
    c2.metric("Baseline CAGR",  f"{mb.get('cagr_pct', 0):+.1f}%")
    c3.metric("Sharpe (filt)",  f"{mf.get('sharpe', 0):.2f}")
    c4.metric("Sortino (filt)", f"{mf.get('sortino', 0):.2f}")
    c5.metric("Max DD (filt)",  f"{mf.get('max_dd_pct', 0):.1f}%")
    c6.metric("Max DD (base)",  f"{mb.get('max_dd_pct', 0):.1f}%")

    eq_f = (1 + rf.fillna(0)).cumprod()
    eq_b = (1 + rb.fillna(0)).cumprod()
    dd_f = (eq_f / eq_f.cummax() - 1) * 100
    dd_b = (eq_b / eq_b.cummax() - 1) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38],
                        subplot_titles=["Equity Curve (log scale)", "Drawdown from HWM"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=df["date"], y=eq_f, name="MA13+stor (filtered)",
                             line=dict(color=C_ORANGE, width=2.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=eq_b, name="Baseline (spread sign)",
                             line=dict(color=C_BLUE, width=1.4, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=dd_f, name="DD filtered",
                             fill="tozeroy", fillcolor="rgba(255,109,0,0.12)",
                             line=dict(color=C_ORANGE, width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=dd_b, name="DD baseline",
                             line=dict(color=C_BLUE, width=1, dash="dot")), row=2, col=1)
    fig.update_yaxes(type="log", ticksuffix="x", title_text="Portfolio value", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", title_text="Drawdown", row=2, col=1)
    fig.update_layout(height=520, hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0),
                      plot_bgcolor=C_BG, paper_bgcolor=C_BG,
                      font=dict(color="#ccc"), margin=dict(l=0, r=0, t=44, b=0))
    fig.update_xaxes(gridcolor=C_GRID); fig.update_yaxes(gridcolor=C_GRID)
    st.plotly_chart(fig, use_container_width=True)


def _annual_bars(df: pd.DataFrame, rf: pd.Series, rb: pd.Series):
    df2 = df.copy()
    df2["rf"] = rf.fillna(0).values; df2["rb"] = rb.fillna(0).values
    df2["year"] = pd.to_datetime(df2["date"]).dt.year
    ann = df2.groupby("year").apply(
        lambda g: pd.Series({
            "filtered": (1 + g["rf"]).prod() - 1,
            "baseline": (1 + g["rb"]).prod() - 1,
        })
    ).reset_index()
    years = ann["year"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=ann["baseline"] * 100, name="Baseline",
                         marker_color=[C_BLUE if v >= 0 else "#1c3a6e" for v in ann["baseline"]],
                         opacity=0.6))
    fig.add_trace(go.Bar(x=years, y=ann["filtered"] * 100, name="MA13+stor",
                         marker_color=[C_ORANGE if v >= 0 else "#6b2e00" for v in ann["filtered"]],
                         opacity=0.88))
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    fig.update_layout(height=310, barmode="group", hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0),
                      plot_bgcolor=C_BG, paper_bgcolor=C_BG,
                      font=dict(color="#ccc"),
                      yaxis=dict(ticksuffix="%", gridcolor=C_GRID),
                      xaxis=dict(gridcolor=C_GRID),
                      margin=dict(l=0, r=0, t=36, b=0))
    st.plotly_chart(fig, use_container_width=True)


def _stats_table(rf: pd.Series, rb: pd.Series, total_yrs: float):
    mf = _compute_metrics_m2m(rf, total_yrs)
    mb = _compute_metrics_m2m(rb, total_yrs)
    # Post-2016
    cutoff = pd.Timestamp("2016-01-01")
    rows_def = [
        ("CAGR %",       "cagr_pct",     "{:+.1f}%"),
        ("Sharpe",       "sharpe",        "{:.2f}"),
        ("Sortino",      "sortino",       "{:.2f}"),
        ("Max DD %",     "max_dd_pct",    "{:.1f}%"),
        ("Win Rate %",   "win_rate_pct",  "{:.1f}%"),
        ("Avg Win %",    "avg_win_pct",   "{:+.3f}%"),
        ("Avg Loss %",   "avg_loss_pct",  "{:+.3f}%"),
        ("Skew",         "skew",          "{:.2f}"),
        ("Weeks",        "weeks",         "{:d}"),
    ]
    def _f(d, k, fmt):
        v = d.get(k)
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
    def _tbl(rf_s, rb_s, yrs):
        mf2 = _compute_metrics_m2m(rf_s, yrs)
        mb2 = _compute_metrics_m2m(rb_s, yrs)
        return pd.DataFrame({
            "MA13+stor": {lbl: _f(mf2, k, fmt) for lbl, k, fmt in rows_def},
            "Baseline":  {lbl: _f(mb2, k, fmt) for lbl, k, fmt in rows_def},
        }).T
    ca, cb = st.columns(2)
    with ca:
        st.caption("**Full period**")
        st.dataframe(_tbl(rf, rb, total_yrs), use_container_width=True)
    # Post-2016 subset — rf/rb are date-indexed so .loc works correctly
    with cb:
        st.caption("**Post-2016**")
        rf_post = rf.loc[rf.index >= cutoff]
        rb_post = rb.loc[rb.index >= cutoff]
        if len(rf_post) > 1:
            post_yrs = (rf_post.index[-1] - rf_post.index[0]).days / 365.25
        else:
            post_yrs = total_yrs
        st.dataframe(_tbl(rf_post, rb_post, post_yrs), use_container_width=True)


# ==============================================================================
# RENDER — SIGNAL HISTORY TABLE
# ==============================================================================

def _recent_signals_table(df: pd.DataFrame, n: int = 26):
    recent = (df[df["price_ma13"].notna()]
              .tail(n).sort_values("date", ascending=False).copy())
    recent["Date"]         = recent["date"].dt.strftime("%Y-%m-%d")
    recent["CL1"]          = recent["CL1"].map("${:.2f}".format)
    recent["CL2"]          = recent["CL2"].map("${:.2f}".format)
    recent["Spread%"]      = recent["spread"].map("{:+.4f}%".format)
    recent["MA13 $"]       = recent["price_ma13"].map("${:.2f}".format)
    recent["Pr<MA13"]      = recent["price_below_ma13"].map(
        lambda v: "✓" if v == 1.0 else "✗")
    recent["St>MA26"]      = recent["stor_above_ma26"].map(
        lambda v: "✓" if v == 1.0 else "✗")
    recent["Raw"]          = recent["raw_signal"]
    recent["Final"]        = recent["filtered_signal"]
    recent["Roll"]         = recent["is_roll"].map(lambda v: "🔄" if v else "")
    cols = ["Date", "CL1", "CL2", "Spread%", "MA13 $",
            "Pr<MA13", "St>MA26", "Raw", "Final", "Roll"]
    st.dataframe(recent[cols].set_index("Date"), use_container_width=True, height=520)


# ==============================================================================
# MAIN TAB RENDER
# ==============================================================================

def render_cl_tab():
    with st.spinner("Loading EIA historical data... (first load ~30 sec)"):
        try:
            prices, stor, df, rf, rb, tf, tb = _load_historical()
        except Exception as e:
            st.error(f"Historical data failed: {e}"); return

    with st.spinner("Fetching live prices (yfinance)..."):
        try:
            sig, stor_live = _load_live()
        except Exception as e:
            st.error(f"Live signal failed: {e}"); sig = None; stor_live = stor

    total_yrs = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    data_thru = df["date"].max().strftime("%Y-%m-%d")
    st.caption(
        f"EIA/yfinance prices through **{data_thru}** | "
        f"Storage through **{stor['date'].max().strftime('%Y-%m-%d')}** | "
        f"Weekly M2M | Live via yfinance"
    )

    if sig: _hero(sig)
    else:   st.error("Live signal unavailable.")

    st.markdown("---"); st.markdown("### Signal Audit")
    if sig: _audit(sig, stor_live)

    st.markdown("---"); st.markdown("### Price, Spread & Filter History")
    lb = st.select_slider(
        "Lookback", options=[26, 52, 104, 260, 520], value=104,
        format_func=lambda x: f"{x // 52}yr" if x >= 52 else f"{x}wk",
        key="cl_lb",
    )
    _charts(df, weeks=lb)

    st.markdown("---"); st.markdown("### Backtest Performance")
    st.caption(
        f"Weekly M2M returns | {df['date'].min().year}–{data_thru[:4]} | "
        f"Filtered vs raw spread-sign baseline"
    )
    _equity_chart(df, rf, rb, total_yrs)

    st.markdown("---"); st.markdown("### Annual Returns")
    _annual_bars(df, rf, rb)

    st.markdown("---"); st.markdown("### Performance Statistics")
    _stats_table(rf, rb, total_yrs)

    st.markdown("---")
    with st.expander("Recent signal history (last 26 weeks)", expanded=False):
        st.caption("Pr<MA13: CL1 below 13-wk MA price | St>MA26: storage above 26-wk MA | Roll 🔄 = roll week")
        _recent_signals_table(df, n=26)

    with st.expander("Recent trades (last 30)", expanded=False):
        t = tf.copy().sort_values("entry_date", ascending=False).head(30)
        t["return_pct"]       = (t["return"] * 100).map("{:+.2f}%".format)
        t["entry_date"]       = pd.to_datetime(t["entry_date"]).dt.strftime("%Y-%m-%d")
        t["exit_date"]        = pd.to_datetime(t["exit_date"]).dt.strftime("%Y-%m-%d")
        t["entry_price"]      = t["entry_price"].map("${:.2f}".format)
        t["exit_price"]       = t["exit_price"].map("${:.2f}".format)
        t["spread_entry"]     = t["spread_entry"].map("{:+.4f}%".format)
        t["CL1_entry"]        = t["CL1_entry"].map("${:.2f}".format)
        t["price_ma13_entry"] = t["price_ma13_entry"].map("${:.2f}".format)
        cols = ["entry_date", "exit_date", "signal", "entry_price", "exit_price",
                "return_pct", "hold_weeks", "exit_type",
                "spread_entry", "CL1_entry", "price_ma13_entry"]
        cols = [c for c in cols if c in t.columns]
        st.dataframe(t[cols].set_index("entry_date"), use_container_width=True, height=420)

    with st.expander("Download CSVs", expanded=False):
        ca, cb = st.columns(2)
        tc = tf.copy()
        tc["entry_date"] = pd.to_datetime(tc["entry_date"]).dt.strftime("%Y-%m-%d")
        tc["exit_date"]  = pd.to_datetime(tc["exit_date"]).dt.strftime("%Y-%m-%d")
        ca.download_button("Trade log (filtered)", tc.to_csv(index=False).encode(),
                           "cl_trades.csv", "text/csv")
        sc2 = df[["date", "CL1", "CL2", "spread", "price_ma13", "raw_signal",
                   "filtered_signal", "is_roll", "storage_kbbl", "stor_ma26",
                   "price_below_ma13", "stor_above_ma26"]].copy()
        sc2["date"] = sc2["date"].dt.strftime("%Y-%m-%d")
        cb.download_button("Weekly signal history", sc2.to_csv(index=False).encode(),
                           "cl_signals.csv", "text/csv")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="CL Roll Dashboard",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.html("""<style>
        .block-container{padding-top:1.5rem;padding-bottom:2rem;}
        [data-testid="stMetricValue"]{font-size:1.25rem;font-weight:700;}
    </style>""")

    st.title("🛢️ CL Roll Yield Dashboard")
    st.caption(
        "NYMEX WTI Crude Oil  ·  EIA Data  ·  "
        "Strategy: Spread sign + MA13 price filter + Stor MA26 filter"
    )

    with st.sidebar:
        st.header("Controls")
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        st.divider()
        st.caption(
            "**Strategy: MA13+stor (full)**\n\n"
            "**Signal (weekly)**\n"
            "- Spread = (CL1−CL2)/CL1×100\n"
            "- LONG  if spread ≥ 0 (backwardation)\n"
            "- SHORT if spread  < 0 (contango)\n\n"
            "**Short filter — BOTH required**\n"
            "- CL1 < 13-week MA of CL1  [trend]\n"
            "- Storage > 26-week MA     [supply]\n"
            "  Longs are unfiltered.\n\n"
            "**Execution**\n"
            "- Entry: C2 price\n"
            "- Roll exit: C1 price (convergence)\n"
            "- Signal exit: C2 price\n"
            "- Roll date: last Fri before expiry\n\n"
            "**Data**\n"
            "- Prices: EIA XLS / yfinance fallback\n"
            "- Storage: EIA API v2 (excl. SPR)\n"
            "- Live: yfinance\n\n"
            "**Backtest vs baseline**\n"
            "CAGR +17.4% vs +14.4%\n"
            "Sharpe 0.69 vs 0.56\n"
            "MaxDD −54% vs −64%"
        )

    render_cl_tab()
