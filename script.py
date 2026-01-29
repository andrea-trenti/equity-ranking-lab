#!/usr/bin/env python3
# script.py
# Multifactor ranking for S&P 500 with robust data handling:
# - Prices/momentum/risk from Yahoo via yfinance
# - Fundamentals primarily from yfinance financial statements (with safer parsing + cache)
# - Optional fallback to SEC EDGAR XBRL "companyfacts" (official) when Yahoo fundamentals missing
# - Sector-neutral scoring with continuous trend and beta computed from returns
#
# Output:
#   sp500_multifactor_ranking.csv

import os
import json
import time
import math
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

# ----------------------------
# Config
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CACHE_DIR = os.path.join(BASE_DIR, "cache_sp500_v5")
os.makedirs(CACHE_DIR, exist_ok=True)

BENCHMARK = "SPY"         # used to compute beta
PRICE_LOOKBACK_YEARS = 2  # download window
TRADING_DAYS = 252

# Minimum amount of usable metrics to keep a ticker in the ranking
MIN_TOTAL_METRICS = 8  # tune: 8–12 recommended depending on data coverage

# Scoring weights (will be renormalized by available categories)
CATEGORY_WEIGHTS = {
    "Growth": 0.20,
    "Quality": 0.20,
    "Value": 0.20,
    "Momentum": 0.25,
    "Risk": 0.15,
}

# SEC EDGAR (official) endpoints
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

# Identify yourself to SEC (official requirement). Put a real email if you want to be safe.
SEC_HEADERS = {
    "User-Agent": "sp500-multifactor-research/1.0 (contact: example@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# SEC rate-limit guidance: no more than ~10 requests/second.
SEC_MIN_INTERVAL_SEC = 0.15

# yfinance retry/backoff
YF_MAX_RETRIES = 4
YF_BASE_SLEEP = 0.8


# ----------------------------
# Utilities
# ----------------------------
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (float, int, np.floating, np.integer)):
            v = float(x)
            if math.isfinite(v):
                return v
            return default
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return default
        v = float(s)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def safe_get(obj, key, default=np.nan):
    """Dict/object-safe getter."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except Exception:
        pass
    return getattr(obj, key, default)


def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Percentile rank to [0, 100]. NaNs remain NaN.
    If higher_is_better=False, ranks are inverted.
    """
    if s.dropna().empty:
        return pd.Series(index=s.index, dtype=float)
    r = s.rank(pct=True, method="average")
    if not higher_is_better:
        r = 1 - r
    return 100 * r


def ensure_cache_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def load_cache(path: str) -> Optional[Any]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_cache(path: str, obj: Any) -> None:
    try:
        ensure_cache_dir(path)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass


def cache_is_fresh(path: str, ttl_seconds: int) -> bool:
    try:
        if not os.path.exists(path):
            return False
        age = time.time() - os.path.getmtime(path)
        return age <= ttl_seconds
    except Exception:
        return False


def normalize_ticker_for_yahoo(t: str) -> str:
    # Yahoo uses '-' instead of '.' for share classes (e.g., BRK-B instead of BRK.B)
    return t.replace(".", "-").upper().strip()


def normalize_ticker_for_sec(t: str) -> str:
    # SEC mapping file uses tickers like "BRK.B" (often). We'll try both.
    return t.upper().strip()


# ----------------------------
# Universe: S&P 500 (Wikipedia + fallbacks)
# ----------------------------
def fetch_sp500_universe(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns DataFrame with columns: Symbol, Security, GICS Sector, GICS Sub-Industry, Symbol_Yahoo.

    Why not pd.read_html(url)?
    - Some sites (including Wikipedia) may return HTTP 403 to generic urllib clients.
    - We therefore fetch HTML via requests with a browser-like User-Agent, then parse locally.

    Fallbacks:
    1) Cached universe (CSV) in CACHE_DIR (to avoid repeated hits / transient blocks).
    2) Wikipedia HTML table (preferred because it includes GICS sub-industry).
    3) Public CSV mirrors (may have fewer columns).
    """
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    fallback_csv_urls = [
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    ]

    cache_csv = cache_path("sp500_universe_cached.csv")
    cache_ttl = 7 * 86400  # refresh weekly

    # 1) cache
    if not force_refresh and cache_is_fresh(cache_csv, cache_ttl):
        try:
            dfc = pd.read_csv(cache_csv)
            need = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "Symbol_Yahoo"}
            if need.issubset(set(dfc.columns)) and len(dfc) >= 450:
                return dfc
        except Exception:
            pass

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # 2) Wikipedia
    try:
        r = requests.get(wiki_url, headers=headers, timeout=30)
        r.raise_for_status()
        html = r.text
        try:
            tables = pd.read_html(html)
        except Exception as e:
            raise RuntimeError(
                "pandas.read_html failed to parse Wikipedia HTML. "
                "Install an HTML parser: pip install lxml html5lib"
            ) from e

        # pick the table that contains Symbol & Security
        t0 = None
        for t in tables:
            cols = set(map(str, t.columns))
            if {"Symbol", "Security"}.issubset(cols):
                t0 = t
                break
        if t0 is None:
            raise RuntimeError("Could not find the S&P 500 table on Wikipedia (page format changed).")

        df = t0.copy()

        keep = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise RuntimeError(f"Wikipedia table missing expected columns: {missing}")

        df = df[keep].copy()
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        df["Symbol_Yahoo"] = df["Symbol"].str.replace(".", "-", regex=False).str.upper().str.strip()

        try:
            df.to_csv(cache_csv, index=False)
        except Exception:
            pass
        return df

    except Exception as wiki_err:
        print(f"[universe] Wikipedia fetch/parse failed: {wiki_err}")

    # 3) CSV mirrors (reduced metadata)
    for u in fallback_csv_urls:
        try:
            r = requests.get(u, headers=headers, timeout=30)
            r.raise_for_status()
            from io import StringIO
            raw = StringIO(r.text)
            t = pd.read_csv(raw)

            # Common schema: Symbol, Name, Sector
            if "Symbol" in t.columns and ("Name" in t.columns or "Security" in t.columns):
                df = pd.DataFrame()
                df["Symbol"] = t["Symbol"].astype(str).str.strip()
                df["Security"] = (t["Security"] if "Security" in t.columns else t["Name"]).astype(str).str.strip()
                if "GICS Sector" in t.columns:
                    df["GICS Sector"] = t["GICS Sector"].astype(str)
                elif "Sector" in t.columns:
                    df["GICS Sector"] = t["Sector"].astype(str)
                else:
                    df["GICS Sector"] = ""

                df["GICS Sub-Industry"] = t["GICS Sub-Industry"].astype(str) if "GICS Sub-Industry" in t.columns else ""
                df["Symbol_Yahoo"] = df["Symbol"].str.replace(".", "-", regex=False).str.upper().str.strip()

                try:
                    df.to_csv(cache_csv, index=False)
                except Exception:
                    pass
                return df
        except Exception as e:
            print(f"[universe] fallback CSV failed ({u}): {e}")

    raise RuntimeError(
        "Failed to load S&P 500 universe from Wikipedia and fallback sources. "
        "Check your network, or provide a cached universe CSV in the cache folder."
    )


# ----------------------------
# Prices & risk/momentum features
# ----------------------------
def download_prices_panel(tickers_yahoo: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns as tickers and rows as dates (Close prices, auto_adjusted).
    """
    all_tickers = sorted(set([t for t in tickers_yahoo if isinstance(t, str) and t] + [BENCHMARK]))
    for attempt in range(1, YF_MAX_RETRIES + 1):
        try:
            data = yf.download(
                tickers=all_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if data is None or data.empty:
                raise RuntimeError("Empty prices from yfinance.")
            if isinstance(data.columns, pd.MultiIndex):
                closes = {}
                for t in all_tickers:
                    if (t, "Close") in data.columns:
                        closes[t] = data[(t, "Close")]
                    elif (t, "Adj Close") in data.columns:
                        closes[t] = data[(t, "Adj Close")]
                panel = pd.DataFrame(closes)
            else:
                col = "Close" if "Close" in data.columns else ("Adj Close" if "Adj Close" in data.columns else None)
                if col is None:
                    raise RuntimeError("No Close/Adj Close in yfinance data.")
                panel = data[[col]].rename(columns={col: all_tickers[0]})
            panel = panel.dropna(axis=1, how="all")
            return panel
        except Exception as e:
            sleep = YF_BASE_SLEEP * (2 ** (attempt - 1)) + random.random() * 0.3
            print(f"[prices] attempt {attempt}/{YF_MAX_RETRIES} failed: {e} -> sleep {sleep:.2f}s")
            time.sleep(sleep)
    raise RuntimeError("Failed to download prices after retries.")


def compute_price_features(price_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price-based features per ticker.
    """
    price_panel = price_panel.sort_index()
    rets = price_panel.pct_change()

    feats = []
    for t in price_panel.columns:
        s = price_panel[t].dropna()
        if len(s) < 260:
            continue

        last = s.iloc[-1]
        r = rets[t].dropna()

        def mom(days):
            if len(s) <= days:
                return np.nan
            return (s.iloc[-1] / s.iloc[-(days + 1)]) - 1

        mom_3m = mom(63)
        mom_6m = mom(126)
        mom_12m = mom(252)
        mom_12_1 = np.nan
        if len(s) > 273:
            mom_12_1 = (s.iloc[-22] / s.iloc[-(252 + 22)]) - 1

        vol = np.nan
        if len(r) >= 63:
            vol = r.tail(63).std() * math.sqrt(TRADING_DAYS)

        dd = np.nan
        if len(s) >= 252:
            w = s.tail(252)
            peak = w.cummax()
            dd = ((w / peak) - 1).min()

        ma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan
        trend_dist = np.nan
        trend_bin = np.nan
        if pd.notna(ma200) and ma200 != 0:
            trend_dist = (last / ma200) - 1
            trend_bin = 1.0 if last > ma200 else 0.0

        feats.append({
            "Symbol_Yahoo": t,
            "Last Price": safe_float(last),
            "Momentum 3M": safe_float(mom_3m),
            "Momentum 6M": safe_float(mom_6m),
            "Momentum 12M": safe_float(mom_12m),
            "Momentum 12-1": safe_float(mom_12_1),
            "Volatility 63D (ann.)": safe_float(vol),
            "Max Drawdown 252D": safe_float(dd),
            "Trend Dist MA200": safe_float(trend_dist),
            "Trend MA200 (bin)": safe_float(trend_bin),
        })

    df = pd.DataFrame(feats).set_index("Symbol_Yahoo")

    if BENCHMARK in price_panel.columns:
        m = rets[BENCHMARK].dropna()
        betas = {}
        for t in df.index:
            if t not in rets.columns:
                betas[t] = np.nan
                continue
            x = rets[t].dropna()
            joined = pd.concat([x, m], axis=1, join="inner").dropna()
            if len(joined) < 126:
                betas[t] = np.nan
                continue
            joined = joined.tail(252)
            ri = joined.iloc[:, 0]
            rm = joined.iloc[:, 1]
            var = rm.var()
            if var == 0 or pd.isna(var):
                betas[t] = np.nan
            else:
                betas[t] = ri.cov(rm) / var
        df["Beta (calc)"] = pd.Series(betas)
    else:
        df["Beta (calc)"] = np.nan
    return df


# ----------------------------
# Fundamentals: yfinance + SEC fallback
# ----------------------------
def _get_statement_df(t: yf.Ticker, attr_names: List[str]) -> Optional[pd.DataFrame]:
    for nm in attr_names:
        try:
            v = getattr(t, nm, None)
            if callable(v):
                v = v()
            if isinstance(v, pd.DataFrame) and not v.empty:
                return v
        except Exception:
            continue
    return None


def _get_line_value(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx = [str(i) for i in df.index]
    idx_l = [i.lower() for i in idx]
    for c in candidates:
        c_l = c.lower()
        if c_l in idx_l:
            pos = idx_l.index(c_l)
            return df.iloc[pos]
    for c in candidates:
        c_l = c.lower()
        matches = [i for i in idx_l if c_l in i]
        if matches:
            pos = idx_l.index(matches[0])
            return df.iloc[pos]
    return None


def _series_from_statement_row(row: pd.Series) -> pd.Series:
    if row is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(row, errors="coerce")


def compute_fundamentals_from_yf(ticker_yahoo: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    t = yf.Ticker(ticker_yahoo)

    try:
        fi = getattr(t, "fast_info", None)
    except Exception:
        fi = None

    out["Market Cap"] = safe_float(safe_get(fi, "market_cap"))
    out["Shares Outstanding"] = safe_float(safe_get(fi, "shares"))
    out["Beta (yahoo)"] = safe_float(safe_get(fi, "beta"))

    inc = _get_statement_df(t, ["income_stmt", "financials"])
    bal = _get_statement_df(t, ["balance_sheet"])
    cf = _get_statement_df(t, ["cashflow"])

    rev_row = _get_line_value(inc, ["Total Revenue", "TotalRevenue", "Revenue", "Revenues"])
    ni_row = _get_line_value(inc, ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeLoss"])
    op_row = _get_line_value(inc, ["Operating Income", "OperatingIncome", "EBIT", "Ebit"])
    ebitda_row = _get_line_value(inc, ["EBITDA", "Ebitda"])

    assets_row = _get_line_value(bal, ["Total Assets", "TotalAssets"])
    equity_row = _get_line_value(bal, ["Total Stockholder Equity", "Total Equity", "Stockholders Equity", "TotalStockholdersEquity"])
    debt_row = _get_line_value(bal, ["Total Debt", "Long Term Debt", "LongTermDebt", "TotalDebt"])
    cash_row = _get_line_value(bal, ["Cash And Cash Equivalents", "Cash", "CashAndCashEquivalentsAtCarryingValue"])

    cfo_row = _get_line_value(cf, ["Total Cash From Operating Activities", "Operating Cash Flow", "NetCashProvidedByUsedInOperatingActivities"])
    capex_row = _get_line_value(cf, ["Capital Expenditures", "CapitalExpenditures", "PaymentsToAcquirePropertyPlantAndEquipment"])

    def latest_from_row(row):
        s = _series_from_statement_row(row).dropna()
        if s.empty:
            return np.nan
        return safe_float(s.iloc[0])

    revenue = latest_from_row(rev_row)
    net_income = latest_from_row(ni_row)
    op_income = latest_from_row(op_row)
    ebitda = latest_from_row(ebitda_row)
    assets = latest_from_row(assets_row)
    equity = latest_from_row(equity_row)
    debt = latest_from_row(debt_row)
    cash = latest_from_row(cash_row)
    cfo = latest_from_row(cfo_row)
    capex = latest_from_row(capex_row)

    out["Revenue (FY)"] = revenue
    out["Net Income (FY)"] = net_income
    out["Operating Income (FY)"] = op_income
    out["EBITDA (FY)"] = ebitda
    out["Total Assets (FY)"] = assets
    out["Total Equity (FY)"] = equity
    out["Total Debt (FY)"] = debt
    out["Cash (FY)"] = cash
    out["CFO (FY)"] = cfo
    out["CapEx (FY)"] = capex

    out["Net Margin"] = safe_float(net_income / revenue) if pd.notna(net_income) and pd.notna(revenue) and revenue != 0 else np.nan
    out["Op Margin"] = safe_float(op_income / revenue) if pd.notna(op_income) and pd.notna(revenue) and revenue != 0 else np.nan
    out["ROA"] = safe_float(net_income / assets) if pd.notna(net_income) and pd.notna(assets) and assets != 0 else np.nan
    out["ROE"] = safe_float(net_income / equity) if pd.notna(net_income) and pd.notna(equity) and equity != 0 else np.nan

    tax_rate = 0.21
    invested_cap = np.nan
    if pd.notna(equity) and pd.notna(debt):
        invested_cap = equity + debt - (cash if pd.notna(cash) else 0.0)
    nopat = safe_float(op_income * (1 - tax_rate)) if pd.notna(op_income) else np.nan
    out["ROIC (approx)"] = safe_float(nopat / invested_cap) if pd.notna(nopat) and pd.notna(invested_cap) and invested_cap != 0 else np.nan

    out["FCF (FY)"] = safe_float(cfo - capex) if pd.notna(cfo) and pd.notna(capex) else np.nan

    rev_series = _series_from_statement_row(rev_row).dropna()
    if len(rev_series) >= 4:
        rev_vals = rev_series.values[:4][::-1]
        if len(rev_vals) == 4 and rev_vals[0] > 0:
            out["Revenue CAGR 3Y"] = safe_float((rev_vals[-1] / rev_vals[0]) ** (1/3) - 1)
        else:
            out["Revenue CAGR 3Y"] = np.nan
    else:
        out["Revenue CAGR 3Y"] = np.nan

    return out


# --- SEC helpers ---
_sec_last_call = 0.0

def _sec_throttle():
    global _sec_last_call
    now = time.time()
    wait = SEC_MIN_INTERVAL_SEC - (now - _sec_last_call)
    if wait > 0:
        time.sleep(wait)
    _sec_last_call = time.time()


def _http_get_json(url: str, headers: Dict[str, str], max_retries: int = 4, base_sleep: float = 0.8) -> Optional[Dict[str, Any]]:
    for attempt in range(1, max_retries + 1):
        try:
            _sec_throttle()
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()
            sleep = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.4
            print(f"[SEC] {url} status {r.status_code} attempt {attempt} -> sleep {sleep:.2f}s")
            time.sleep(sleep)
        except Exception as e:
            sleep = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.4
            print(f"[SEC] {url} error {e} attempt {attempt} -> sleep {sleep:.2f}s")
            time.sleep(sleep)
    return None


def load_sec_ticker_map(force_refresh: bool = False) -> Dict[str, int]:
    path = cache_path("sec_company_tickers.pkl")
    ttl = 30 * 86400
    if not force_refresh and cache_is_fresh(path, ttl):
        obj = load_cache(path)
        if isinstance(obj, dict) and obj:
            return obj

    data = _http_get_json(SEC_TICKER_MAP_URL, headers=SEC_HEADERS)
    mapping: Dict[str, int] = {}
    if isinstance(data, dict):
        for _, row in data.items():
            try:
                tick = str(row.get("ticker", "")).upper()
                cik = int(row.get("cik_str"))
                if tick:
                    mapping[tick] = cik
            except Exception:
                continue
    if mapping:
        save_cache(path, mapping)
    return mapping


def sec_companyfacts_metrics(ticker_raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}

    sec_map = load_sec_ticker_map()
    tsec = normalize_ticker_for_sec(ticker_raw)
    cik = sec_map.get(tsec)
    if cik is None:
        alt = tsec.replace("-", ".")
        cik = sec_map.get(alt)
    if cik is None:
        return out

    cik10 = str(cik).zfill(10)
    url = SEC_COMPANYFACTS_URL.format(cik10=cik10)
    js = _http_get_json(url, headers=SEC_HEADERS)
    if not js:
        return out

    facts = js.get("facts", {})
    usgaap = facts.get("us-gaap", {}) if isinstance(facts, dict) else {}

    def pick_latest_annual(tag_names: List[str], unit_preference: List[str]) -> float:
        for tag in tag_names:
            node = usgaap.get(tag)
            if not isinstance(node, dict):
                continue
            units = node.get("units", {})
            if not isinstance(units, dict):
                continue
            for u in unit_preference:
                arr = units.get(u)
                if isinstance(arr, list) and arr:
                    df = pd.DataFrame(arr)
                    if "fp" in df.columns:
                        fy = df[df["fp"].astype(str).str.upper().eq("FY")]
                        if not fy.empty:
                            fy = fy.sort_values("end", ascending=False)
                            return safe_float(fy.iloc[0].get("val"))
                    if "form" in df.columns:
                        k = df[df["form"].astype(str).str.upper().eq("10-K")]
                        if not k.empty:
                            k = k.sort_values("end", ascending=False)
                            return safe_float(k.iloc[0].get("val"))
                    df = df.sort_values("end", ascending=False)
                    return safe_float(df.iloc[0].get("val"))
        return np.nan

    out["Revenue (FY)"] = pick_latest_annual(["Revenues", "SalesRevenueNet"], ["USD"])
    out["Net Income (FY)"] = pick_latest_annual(["NetIncomeLoss"], ["USD"])
    out["Total Assets (FY)"] = pick_latest_annual(["Assets"], ["USD"])
    out["Total Equity (FY)"] = pick_latest_annual(
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], ["USD"]
    )
    out["CFO (FY)"] = pick_latest_annual(["NetCashProvidedByUsedInOperatingActivities"], ["USD"])
    out["CapEx (FY)"] = pick_latest_annual(["PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpenditures"], ["USD"])
    out["Shares Outstanding"] = pick_latest_annual(["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"], ["shares"])
    return out


# ----------------------------
# Fundamentals fetch with cache + quality-based TTL
# ----------------------------
def fundamentals_for_ticker(symbol_raw: str, symbol_yahoo: str) -> Dict[str, float]:
    cpath = cache_path(f"fund_{symbol_yahoo}.pkl")

    cached = load_cache(cpath)
    if isinstance(cached, dict) and "data" in cached and "quality" in cached and "saved_at" in cached:
        quality = int(cached.get("quality", 0))
        ttl = (14 * 86400) if quality >= 6 else (6 * 3600)
        if cache_is_fresh(cpath, ttl):
            return cached["data"]

    data = {}
    quality = 0
    try:
        data = compute_fundamentals_from_yf(symbol_yahoo)
        keyset = [
            "Market Cap", "Revenue (FY)", "Net Income (FY)", "Total Assets (FY)", "Total Equity (FY)",
            "CFO (FY)", "CapEx (FY)", "FCF (FY)", "Revenue CAGR 3Y", "ROIC (approx)"
        ]
        quality = sum(1 for k in keyset if pd.notna(data.get(k, np.nan)))
    except Exception as e:
        print(f"[fund-yf] {symbol_yahoo} error: {e}")
        data = {}
        quality = 0

    if quality < 4:
        try:
            secd = sec_companyfacts_metrics(symbol_raw)
            for k, v in secd.items():
                if k not in data or pd.isna(data.get(k)):
                    data[k] = v

            revenue = data.get("Revenue (FY)", np.nan)
            ni = data.get("Net Income (FY)", np.nan)
            assets = data.get("Total Assets (FY)", np.nan)
            equity = data.get("Total Equity (FY)", np.nan)
            cfo = data.get("CFO (FY)", np.nan)
            capex = data.get("CapEx (FY)", np.nan)

            if pd.notna(ni) and pd.notna(revenue) and revenue != 0:
                data["Net Margin"] = safe_float(ni / revenue)
            if pd.notna(ni) and pd.notna(assets) and assets != 0:
                data["ROA"] = safe_float(ni / assets)
            if pd.notna(ni) and pd.notna(equity) and equity != 0:
                data["ROE"] = safe_float(ni / equity)
            if pd.notna(cfo) and pd.notna(capex):
                data["FCF (FY)"] = safe_float(cfo - capex)
        except Exception as e:
            print(f"[fund-sec] {symbol_raw} error: {e}")

        keyset2 = [
            "Market Cap", "Shares Outstanding", "Revenue (FY)", "Net Income (FY)", "Total Assets (FY)", "Total Equity (FY)",
            "CFO (FY)", "CapEx (FY)", "FCF (FY)", "ROE", "ROA"
        ]
        quality = sum(1 for k in keyset2 if pd.notna(data.get(k, np.nan)))

    save_cache(cpath, {"saved_at": time.time(), "quality": int(quality), "data": data})
    return data


# ----------------------------
# Scoring
# ----------------------------
@dataclass
class MetricDef:
    name: str
    higher_is_better: bool
    category: str


METRICS: List[MetricDef] = [
    MetricDef("Revenue CAGR 3Y", True, "Growth"),
    MetricDef("ROIC (approx)", True, "Quality"),
    MetricDef("ROE", True, "Quality"),
    MetricDef("ROA", True, "Quality"),
    MetricDef("Net Margin", True, "Quality"),
    MetricDef("Op Margin", True, "Quality"),
    MetricDef("P/E (approx)", False, "Value"),
    MetricDef("P/S (approx)", False, "Value"),
    MetricDef("FCF Yield (approx)", True, "Value"),
    MetricDef("Momentum 12-1", True, "Momentum"),
    MetricDef("Momentum 6M", True, "Momentum"),
    MetricDef("Momentum 3M", True, "Momentum"),
    MetricDef("Trend Dist MA200", True, "Momentum"),
    MetricDef("Volatility 63D (ann.)", False, "Risk"),
    MetricDef("Max Drawdown 252D", True, "Risk"),
    MetricDef("Beta", False, "Risk"),
]


def compute_derived_valuation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Market Cap" not in out.columns:
        out["Market Cap"] = np.nan

    out["Shares Outstanding"] = out.get("Shares Outstanding", np.nan)
    mcap = out["Market Cap"]
    price = out["Last Price"]
    missing_sh = out["Shares Outstanding"].isna() & mcap.notna() & price.notna() & (price != 0)
    out.loc[missing_sh, "Shares Outstanding"] = (mcap[missing_sh] / price[missing_sh])

    rev = out.get("Revenue (FY)", np.nan)
    ni = out.get("Net Income (FY)", np.nan)
    fcf = out.get("FCF (FY)", np.nan)

    out["P/S (approx)"] = np.where(mcap.notna() & rev.notna() & (rev != 0), mcap / rev, np.nan)
    out["P/E (approx)"] = np.where(mcap.notna() & ni.notna() & (ni != 0), mcap / ni, np.nan)
    out["FCF Yield (approx)"] = np.where(mcap.notna() & fcf.notna() & (mcap != 0), fcf / mcap, np.nan)
    return out


def score_metrics(df: pd.DataFrame, sector_col: str = "GICS Sector", sector_neutral: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = df.copy()
    for m in METRICS:
        if m.name in base.columns:
            base[m.name] = pd.to_numeric(base[m.name], errors="coerce")
            base[m.name] = winsorize_series(base[m.name], p=0.01)

    metric_scores = pd.DataFrame(index=base.index)

    def score_group(group: pd.DataFrame, metric: MetricDef) -> pd.Series:
        return pct_rank(group[metric.name], higher_is_better=metric.higher_is_better)

    if sector_neutral and sector_col in base.columns:
        for metric in METRICS:
            if metric.name not in base.columns:
                continue
            metric_scores[metric.name] = base.groupby(sector_col, group_keys=False).apply(lambda g: score_group(g, metric))
    else:
        for metric in METRICS:
            if metric.name not in base.columns:
                continue
            metric_scores[metric.name] = pct_rank(base[metric.name], higher_is_better=metric.higher_is_better)

    category_scores = pd.DataFrame(index=base.index)
    for cat in CATEGORY_WEIGHTS.keys():
        cols = [m.name for m in METRICS if m.category == cat and m.name in metric_scores.columns]
        category_scores[cat] = metric_scores[cols].mean(axis=1, skipna=True) if cols else np.nan
    return metric_scores, category_scores


def compute_total_score(category_scores: pd.DataFrame, metric_scores: pd.DataFrame) -> pd.Series:
    weights = pd.Series(CATEGORY_WEIGHTS, dtype=float)

    avail = category_scores.notna()
    w_mat = pd.DataFrame(np.tile(weights.values, (len(category_scores), 1)), index=category_scores.index, columns=weights.index)
    w_mat = w_mat.where(avail, 0.0)
    w_sum = w_mat.sum(axis=1).replace(0, np.nan)
    w_norm = w_mat.div(w_sum, axis=0)

    raw = (category_scores * w_norm).sum(axis=1, skipna=True)

    total_metrics = len([m for m in METRICS if m.name in metric_scores.columns])
    present = metric_scores.notna().sum(axis=1)
    missing_ratio = 1 - (present / max(total_metrics, 1))
    penalty = 20.0 * missing_ratio
    return (raw - penalty).clip(lower=0, upper=100)


def assign_quantile_rating(score: pd.Series) -> pd.Series:
    ranks = score.rank(pct=True, method="average")
    def grade(p):
        if pd.isna(p): return "NA"
        if p >= 0.90: return "A"
        if p >= 0.70: return "B"
        if p >= 0.40: return "C"
        if p >= 0.10: return "D"
        return "F"
    return ranks.map(grade)


def print_diagnostics(df: pd.DataFrame, metric_scores: pd.DataFrame, category_scores: pd.DataFrame, total: pd.Series):
    n = len(df)
    print("\n=== DIAGNOSTICS ===")
    print(f"Universe size: {n}")
    cols_to_check = ["Market Cap", "Revenue (FY)", "Net Income (FY)", "ROE", "ROIC (approx)", "P/E (approx)",
                     "Momentum 12-1", "Volatility 63D (ann.)", "Beta"]
    for c in cols_to_check:
        if c in df.columns:
            cov = df[c].notna().mean() * 100
            print(f"Coverage {c:>22}: {cov:6.1f}%")
    print("Category coverage:")
    for c in category_scores.columns:
        cov = category_scores[c].notna().mean() * 100
        print(f"  {c:>8}: {cov:6.1f}%")
    print(f"Total score range: min={total.min():.2f}, mean={total.mean():.2f}, max={total.max():.2f}")
    print("Ratings counts:")
    print(assign_quantile_rating(total).value_counts(dropna=False))


# ----------------------------
# Main
# ----------------------------
def main():
    uni = fetch_sp500_universe().set_index("Symbol_Yahoo")

    end = datetime.utcnow().date()
    start = (end - timedelta(days=int(365.25 * PRICE_LOOKBACK_YEARS))).isoformat()
    end_s = end.isoformat()

    print(f"[universe] tickers: {len(uni)}")
    print(f"[prices] downloading {PRICE_LOOKBACK_YEARS}y from {start} to {end_s} ...")
    price_panel = download_prices_panel(list(uni.index), start=start, end=end_s)

    pf = compute_price_features(price_panel)

    rows = []
    for sym_yahoo in pf.index:
        if sym_yahoo in uni.index:
            raw_symbol = uni.loc[sym_yahoo, "Symbol"]
        else:
            raw_symbol = sym_yahoo.replace("-", ".")
        fund = fundamentals_for_ticker(str(raw_symbol), str(sym_yahoo))
        rows.append((sym_yahoo, fund))
    fund_df = pd.DataFrame({k: v for k, v in rows}).T
    fund_df.index.name = "Symbol_Yahoo"

    df = uni.join(pf, how="left").join(fund_df, how="left")

    df["Beta"] = df.get("Beta (calc)", np.nan)
    if "Beta" in df.columns and "Beta (yahoo)" in df.columns:
        df["Beta"] = df["Beta"].fillna(df["Beta (yahoo)"])

    df = compute_derived_valuation(df)

    numeric_cols = [m.name for m in METRICS if m.name in df.columns]
    df["Metrics Present"] = df[numeric_cols].notna().sum(axis=1)
    df = df[df["Metrics Present"] >= MIN_TOTAL_METRICS].copy()
    print(f"[filter] kept {len(df)} tickers with >= {MIN_TOTAL_METRICS} metrics present (out of {len(uni)})")

    metric_scores, category_scores = score_metrics(df, sector_col="GICS Sector", sector_neutral=True)
    total = compute_total_score(category_scores, metric_scores)

    df["Total Score"] = total
    df["Rating"] = assign_quantile_rating(total)

    for c in category_scores.columns:
        df[f"Score {c}"] = category_scores[c]
    for m in ["Revenue CAGR 3Y", "ROIC (approx)", "P/E (approx)", "Momentum 12-1", "Volatility 63D (ann.)", "Beta"]:
        if m in metric_scores.columns:
            df[f"Score {m}"] = metric_scores[m]

    df = df.sort_values("Total Score", ascending=False)
    print_diagnostics(df, metric_scores, category_scores, total)

    out_path = os.path.join(BASE_DIR, "sp500_multifactor_ranking.csv")
    df.to_csv(out_path, index=True)
    print(f"\n[output] saved: {out_path}")


if __name__ == "__main__":
    main()
