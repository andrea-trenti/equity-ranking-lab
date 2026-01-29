#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S&P 500 "LITE" multifactor ranking (high coverage)
- Uses: price history + fast_info + fallback to get_info()
- Avoids: income_stmt / balance_sheet / cashflow (often empty / rate-limited)
- Handles: 429/timeout with retries + exponential backoff
- Output: sp500_multifactor_ranking_lite.csv  (same name as before + "_lite")
"""

import time
import math
import random
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from io import StringIO

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# ---- rate / retry knobs
SLEEP_BETWEEN_TICKERS_SEC = 0.15      # small base sleep; backoff handles 429
MAX_RETRIES = 4
BACKOFF_BASE_SEC = 1.2               # exponential backoff base
JITTER_SEC = 0.35                    # random jitter added to sleeps
HISTORY_YEARS = 6                    # enough for 5Y metrics
HISTORY_INTERVAL = "1d"

OUT_CSV = "sp500_multifactor_ranking_lite.csv"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def pct_return(s: pd.Series, days: int):
    """Return over the last N trading days using close series."""
    if s is None or len(s) < days + 1:
        return np.nan
    a = s.iloc[-(days + 1)]
    b = s.iloc[-1]
    if a == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (b / a) - 1.0

def cagr(s: pd.Series, years: float):
    """CAGR from close series, using ~252 trading days/year."""
    if s is None:
        return np.nan
    n = int(round(252 * years))
    if len(s) < n + 1:
        return np.nan
    a = s.iloc[-(n + 1)]
    b = s.iloc[-1]
    if a <= 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (b / a) ** (1.0 / years) - 1.0

def annualized_volatility(close: pd.Series):
    """Annualized vol from daily log returns."""
    if close is None or len(close) < 60:
        return np.nan
    r = np.log(close).diff().dropna()
    if len(r) < 30:
        return np.nan
    return float(r.std() * math.sqrt(252))

def max_drawdown(close: pd.Series):
    """Max drawdown from close series (negative number)."""
    if close is None or len(close) < 30:
        return np.nan
    cummax = close.cummax()
    dd = (close / cummax) - 1.0
    return float(dd.min())

def winsorize_series(x: pd.Series, lower_q=0.05, upper_q=0.95):
    """Clip extremes to reduce outlier impact."""
    x = x.copy()
    vals = x.dropna()
    if vals.empty:
        return x
    lo = vals.quantile(lower_q)
    hi = vals.quantile(upper_q)
    return x.clip(lower=lo, upper=hi)

def percentile_score(series: pd.Series, higher_is_better=True):
    """
    Convert a numeric series to 0-100 percentile scores.
    NaN stays NaN; can be averaged later with skipna.
    """
    s = series.copy()
    # percentile rank: 0..1
    ranks = s.rank(pct=True, method="average")
    if not higher_is_better:
        ranks = 1.0 - ranks
    return ranks * 100.0

def letter_grade(score):
    if pd.isna(score):
        return ""
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"

# ---------------------------------------------------------------------
# 1) Load S&P 500 tickers
# ---------------------------------------------------------------------
resp = requests.get(URL, headers=HEADERS, timeout=20)
resp.raise_for_status()

# pandas FutureWarning fix: wrap html in StringIO
sp500 = pd.read_html(StringIO(resp.text))[0]

tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()

company_by_ticker = (
    sp500.assign(Symbol=sp500["Symbol"].str.replace(".", "-", regex=False))
    .set_index("Symbol")["Security"]
    .to_dict()
)

sector_by_ticker = (
    sp500.assign(Symbol=sp500["Symbol"].str.replace(".", "-", regex=False))
    .set_index("Symbol")["GICS Sector"]
    .to_dict()
)

print(f"Ticker S&P 500: {len(tickers)}")

# ---------------------------------------------------------------------
# 2) Fetch "lite" data for ALL tickers (no 429 cap)
# ---------------------------------------------------------------------
rows = []

for i, tk in enumerate(tickers, start=1):
    print(f"[{i}/{len(tickers)}] Processing: {tk}")

    # retry loop
    ok = False
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = yf.Ticker(tk)

            # ---- history (primary source for lite metrics)
            hist = t.history(
                period=f"{HISTORY_YEARS}y",
                interval=HISTORY_INTERVAL,
                auto_adjust=False,
                actions=False,
                prepost=False,
            )

            if hist is None or hist.empty or "Close" not in hist.columns:
                raise RuntimeError("Empty history")

            close = hist["Close"].dropna()
            if close.empty or len(close) < 60:
                raise RuntimeError("Not enough history")

            price = float(close.iloc[-1])

            # ---- fast_info (fast, often available)
            fi = getattr(t, "fast_info", None)
            fi = fi if isinstance(fi, dict) else {}

            # Fallback for missing values: get_info() (slower, more likely to 429)
            # We ONLY call it if needed.
            info = {}

            def need_info(keys):
                for k in keys:
                    v = fi.get(k, None)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return True
                return False

            # fields we might need for scoring
            if need_info(["beta", "market_cap", "pe_ratio", "last_price", "shares"]):
                # yfinance versions differ: prefer get_info() over .info in newer versions
                info = t.get_info() or {}

            # ---- core fields with fallbacks
            market_cap = safe_float(fi.get("market_cap", info.get("marketCap")))
            beta = safe_float(fi.get("beta", info.get("beta")))
            pe = safe_float(fi.get("pe_ratio", info.get("trailingPE")))
            fwd_pe = safe_float(info.get("forwardPE"))
            ps = safe_float(info.get("priceToSalesTrailing12Months"))
            div_yield = safe_float(info.get("dividendYield"))  # fraction (0.02 = 2%)
            avg_vol = safe_float(fi.get("ten_day_average_volume", info.get("averageVolume")))
            # 52w
            week52_high = safe_float(fi.get("year_high", info.get("fiftyTwoWeekHigh")))
            week52_low = safe_float(fi.get("year_low", info.get("fiftyTwoWeekLow")))

            # ---- price-derived metrics
            ret_1m = pct_return(close, 21)
            ret_3m = pct_return(close, 63)
            ret_6m = pct_return(close, 126)
            ret_1y = pct_return(close, 252)
            cagr_3y = cagr(close, 3.0)
            cagr_5y = cagr(close, 5.0)
            vol_ann = annualized_volatility(close)
            mdd = max_drawdown(close)

            # ---- liquidity proxy (dollar volume)
            dollar_vol = price * avg_vol if (not pd.isna(price) and not pd.isna(avg_vol)) else np.nan

            # ---- distance from 52w high/low
            pct_from_52w_high = (price / week52_high - 1.0) if (week52_high and not pd.isna(week52_high)) else np.nan
            pct_above_52w_low = (price / week52_low - 1.0) if (week52_low and not pd.isna(week52_low)) else np.nan

            rows.append({
                "Ticker": tk,
                "Company": company_by_ticker.get(tk, ""),
                "Sector": sector_by_ticker.get(tk, ""),
                "Market Cap": market_cap,
                "Price": price,
                "Beta": beta,
                "P/E": pe,
                "Forward P/E": fwd_pe,
                "P/S": ps,
                "Dividend Yield": div_yield,
                "Avg Volume": avg_vol,
                "Dollar Volume": dollar_vol,
                "52W High": week52_high,
                "52W Low": week52_low,
                "Pct From 52W High": pct_from_52w_high,
                "Pct Above 52W Low": pct_above_52w_low,
                "Return 1M": ret_1m,
                "Return 3M": ret_3m,
                "Return 6M": ret_6m,
                "Return 1Y": ret_1y,
                "Price CAGR 3Y": cagr_3y,
                "Price CAGR 5Y": cagr_5y,
                "Volatility (ann)": vol_ann,
                "Max Drawdown": mdd,
            })

            ok = True
            break

        except Exception as e:
            last_err = e
            # exponential backoff + jitter
            sleep_s = (BACKOFF_BASE_SEC ** attempt) + random.uniform(0, JITTER_SEC)
            time.sleep(sleep_s)

    if not ok:
        # keep a placeholder row so you can see missing tickers too
        rows.append({
            "Ticker": tk,
            "Company": company_by_ticker.get(tk, ""),
            "Sector": sector_by_ticker.get(tk, ""),
            "Market Cap": np.nan,
            "Price": np.nan,
            "Beta": np.nan,
            "P/E": np.nan,
            "Forward P/E": np.nan,
            "P/S": np.nan,
            "Dividend Yield": np.nan,
            "Avg Volume": np.nan,
            "Dollar Volume": np.nan,
            "52W High": np.nan,
            "52W Low": np.nan,
            "Pct From 52W High": np.nan,
            "Pct Above 52W Low": np.nan,
            "Return 1M": np.nan,
            "Return 3M": np.nan,
            "Return 6M": np.nan,
            "Return 1Y": np.nan,
            "Price CAGR 3Y": np.nan,
            "Price CAGR 5Y": np.nan,
            "Volatility (ann)": np.nan,
            "Max Drawdown": np.nan,
            "Error": str(last_err),
        })

    time.sleep(SLEEP_BETWEEN_TICKERS_SEC + random.uniform(0, JITTER_SEC))

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------
# 3) Scoring (5 categories, percentile based, NaN-tolerant)
# ---------------------------------------------------------------------
# Winsorize the raw metrics to reduce extreme distortions
for col in [
    "Return 1M", "Return 3M", "Return 6M", "Return 1Y",
    "Price CAGR 3Y", "Price CAGR 5Y",
    "P/E", "Forward P/E", "P/S",
    "Beta", "Volatility (ann)", "Max Drawdown",
    "Dividend Yield", "Market Cap", "Dollar Volume"
]:
    if col in df.columns:
        df[col] = winsorize_series(df[col], 0.05, 0.95)

# ---- Category scores (0-100)
# Growth/Momentum: higher returns and higher price CAGRs are better
growth_components = ["Return 1Y", "Return 6M", "Return 3M", "Price CAGR 5Y", "Price CAGR 3Y"]
growth_scores = [percentile_score(df[c], True) for c in growth_components if c in df.columns]
df["Growth Score"] = pd.concat(growth_scores, axis=1).mean(axis=1, skipna=True)

# Profitability (lite proxy): dividend yield + proximity to 52W high (momentum quality-ish)
# If dividend yield missing for many, it won't break; it just gets skipped.
profit_components = ["Dividend Yield", "Pct From 52W High"]
profit_scores = []
if "Dividend Yield" in df.columns:
    profit_scores.append(percentile_score(df["Dividend Yield"], True))
if "Pct From 52W High" in df.columns:
    # closer to 52W high is better -> higher pct_from_high is "less negative / more positive"
    profit_scores.append(percentile_score(df["Pct From 52W High"], True))
df["Profitability Score"] = pd.concat(profit_scores, axis=1).mean(axis=1, skipna=True)

# Quality (lite proxy): large/liquid + stable-ish (use Market Cap + Dollar Volume + 52W range position)
quality_components = []
quality_scores = []
if "Market Cap" in df.columns:
    quality_scores.append(percentile_score(df["Market Cap"], True))
if "Dollar Volume" in df.columns:
    quality_scores.append(percentile_score(df["Dollar Volume"], True))
if "Pct Above 52W Low" in df.columns:
    quality_scores.append(percentile_score(df["Pct Above 52W Low"], True))
df["Quality Score"] = pd.concat(quality_scores, axis=1).mean(axis=1, skipna=True)

# Valuation: lower is better (P/E, Forward P/E, P/S)
val_components = ["P/E", "Forward P/E", "P/S"]
val_scores = [percentile_score(df[c], higher_is_better=False) for c in val_components if c in df.columns]
df["Valuation Score"] = pd.concat(val_scores, axis=1).mean(axis=1, skipna=True)

# Risk: lower beta, lower volatility, less severe max drawdown are better
risk_components = []
risk_scores = []
if "Beta" in df.columns:
    risk_scores.append(percentile_score(df["Beta"], higher_is_better=False))
if "Volatility (ann)" in df.columns:
    risk_scores.append(percentile_score(df["Volatility (ann)"], higher_is_better=False))
if "Max Drawdown" in df.columns:
    # Max drawdown is negative; less negative (closer to 0) is better
    risk_scores.append(percentile_score(df["Max Drawdown"], higher_is_better=True))
df["Risk Score"] = pd.concat(risk_scores, axis=1).mean(axis=1, skipna=True)

# ---- Total score: equal weights (you can tweak later)
df["Total Score"] = df[["Growth Score", "Profitability Score", "Quality Score", "Valuation Score", "Risk Score"]].mean(axis=1, skipna=True)
df["Rating"] = df["Total Score"].apply(letter_grade)

# ---------------------------------------------------------------------
# 4) Output: order + print top 30 + save CSV
# ---------------------------------------------------------------------
# Sort: best total score first; keep missing scores at bottom
df_sorted = df.sort_values(["Total Score", "Market Cap"], ascending=[False, False], na_position="last")

print("\nTOP 30 (Total Score):")
cols_show = [
    "Ticker", "Company", "Total Score", "Rating",
    "Growth Score", "Profitability Score", "Quality Score", "Valuation Score", "Risk Score",
    "Market Cap", "Price", "Beta", "P/E", "Forward P/E", "P/S",
    "Dividend Yield", "Return 1Y", "Price CAGR 5Y", "Volatility (ann)", "Max Drawdown"
]
cols_show = [c for c in cols_show if c in df_sorted.columns]
print(df_sorted[cols_show].head(30).to_string(index=False))

df_sorted.to_csv(OUT_CSV, index=False)
print(f"\nSalvato: {OUT_CSV} | righe: {len(df_sorted)}")
print("\nNOTE: Se vedi ancora rate-limit/429, alza SLEEP_BETWEEN_TICKERS_SEC (es. 0.5–1.2) e/o MAX_RETRIES (es. 6).")
