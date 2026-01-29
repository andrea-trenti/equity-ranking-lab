#!/usr/bin/env python3
"""
finviz_scoring.py
Reads a Finviz-style CSV export and outputs a scored CSV with 6 pillar scores + final score.

Design goals:
- Robust parsing (% values, blanks, '-', numeric already parsed)
- Sector-neutral percentile scoring (if 'Sector' exists), otherwise global
- Uses only columns present; gracefully degrades with missing data
- Emphasizes "regularity" via Stability/Regularity pillar + extra penalty

Usage:
  python finviz_scoring.py --input finviz.csv --output scored.csv
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Parsing helpers
# -----------------------------
_MKT_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

def parse_numeric(x) -> float:
    """Parse numbers from Finviz export:
    - '12.3%' -> 12.3
    - '1.2B' -> 1.2e9
    - '-' / '' -> NaN
    - numbers already numeric -> float
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if s in {"", "-", "—", "N/A", "nan", "None"}:
        return np.nan

    # percent
    if s.endswith("%"):
        try:
            return float(s[:-1].replace(",", ""))
        except ValueError:
            return np.nan

    # unit suffix
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)([KMBT])", s, flags=re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            mult = _MKT_MULT[m.group(2).upper()]
            return val * mult
        except Exception:
            return np.nan

    # plain float with commas
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return np.nan


def winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    """Clip extremes to reduce outlier dominance (robust scoring)."""
    if s.dropna().empty:
        return s
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def pct_rank(
    s: pd.Series,
    higher_is_better: bool = True,
    group: Optional[pd.Series] = None
) -> pd.Series:
    """Percentile rank -> 0..100. Sector-neutral if group provided."""
    # Use 'average' rank to reduce ties sensitivity
    if group is not None:
        r = s.groupby(group).rank(pct=True, method="average")
    else:
        r = s.rank(pct=True, method="average")
    score = r * 100.0
    if not higher_is_better:
        score = 100.0 - score
    return score


def safe_weighted_mean(values: List[pd.Series], weights: List[float]) -> pd.Series:
    """Row-wise weighted mean ignoring NaNs (renormalize weights by availability)."""
    if not values:
        return pd.Series(dtype="float64")
    mat = np.vstack([v.to_numpy(dtype=float) for v in values]).T  # shape (n, k)
    w = np.array(weights, dtype=float)
    # Mask NaNs
    mask = ~np.isnan(mat)
    w_mat = mask * w  # broadcast
    denom = w_mat.sum(axis=1)
    numer = np.nansum(mat * w, axis=1)
    out = np.where(denom > 0, numer / denom, np.nan)
    return pd.Series(out, index=values[0].index)


def rsi_score_linear(rsi: pd.Series) -> pd.Series:
    """Map RSI to a 'stability-friendly' score: ideal ~50, extremes penalized."""
    # 50 -> 100; 0/100 -> 0
    return 100.0 - (rsi - 50.0).abs() * (100.0 / 50.0)


# -----------------------------
# Scoring specs
# -----------------------------
@dataclass(frozen=True)
class MetricSpec:
    col: str
    higher_is_better: bool
    weight: float


PILLARS: Dict[str, List[MetricSpec]] = {
    "Valuation": [
        MetricSpec("P/E", False, 1.0),
        MetricSpec("Forward P/E", False, 1.0),
        MetricSpec("PEG", False, 0.8),
        MetricSpec("P/S", False, 0.8),
        MetricSpec("P/B", False, 0.6),
        MetricSpec("P/Cash", False, 0.4),
        MetricSpec("P/Free Cash Flow", False, 0.8),
        MetricSpec("EV/EBITDA", False, 0.9),
        MetricSpec("EV/Sales", False, 0.7),
    ],
    "Quality": [
        MetricSpec("Return on Assets", True, 0.9),
        MetricSpec("Return on Equity", True, 0.8),
        MetricSpec("Return on Invested Capital", True, 1.0),
        MetricSpec("Gross Margin", True, 0.7),
        MetricSpec("Operating Margin", True, 0.8),
        MetricSpec("Profit Margin", True, 0.9),
        MetricSpec("Current Ratio", True, 0.6),
        MetricSpec("Quick Ratio", True, 0.6),
        MetricSpec("LT Debt/Equity", False, 0.7),
        MetricSpec("Total Debt/Equity", False, 0.9),
    ],
    "Growth": [
        MetricSpec("EPS Growth Past 3 Years", True, 0.8),
        MetricSpec("EPS Growth Past 5 Years", True, 0.8),
        MetricSpec("EPS Growth Next Year", True, 0.6),
        MetricSpec("EPS Growth Next 5 Years", True, 0.9),
        MetricSpec("Sales Growth Past 3 Years", True, 0.8),
        MetricSpec("Sales Growth Past 5 Years", True, 0.8),
        MetricSpec("Sales Growth Quarter Over Quarter", True, 0.6),
        MetricSpec("EPS Growth Quarter Over Quarter", True, 0.6),
        MetricSpec("EPS Year Over Year TTM", True, 0.6),
        MetricSpec("Sales Year Over Year TTM", True, 0.6),
    ],
    "Momentum": [
        # For ETFs/funds, the "Return X Year" columns can substitute for stock price momentum.
        MetricSpec("Return 1 Year", True, 0.6),
        MetricSpec("Return 3 Year", True, 0.6),
        MetricSpec("Return 5 Year", True, 0.6),
        MetricSpec("Performance (Month)", True, 0.7),
        MetricSpec("Performance (Quarter)", True, 0.9),
        MetricSpec("Performance (Half Year)", True, 0.9),
        MetricSpec("Performance (Year)", True, 1.0),
        MetricSpec("20-Day Simple Moving Average", True, 0.5),
        MetricSpec("50-Day Simple Moving Average", True, 0.7),
        MetricSpec("200-Day Simple Moving Average", True, 0.8),
        MetricSpec("52-Week High", True, 0.5),
        MetricSpec("52-Week Low", True, 0.3),
    ],
    "Stability": [
        MetricSpec("Volatility (Week)", False, 1.0),
        MetricSpec("Volatility (Month)", False, 1.0),
        MetricSpec("Beta", False, 0.7),
        MetricSpec("Average True Range", False, 0.5),
        MetricSpec("Relative Strength Index (14)", True, 0.4),  # will be transformed
    ],
    "Sentiment": [
        # ETF demand/flows & cost (if present)
        MetricSpec("Net Expense Ratio", False, 0.6),
        MetricSpec("Assets Under Management", True, 0.4),
        MetricSpec("Net Flows % (1 Month)", True, 0.5),
        MetricSpec("Net Flows % (3 Month)", True, 0.5),
        MetricSpec("Net Flows % (YTD)", True, 0.5),
        MetricSpec("Analyst Recom", False, 0.8),  # 1=Strong Buy, 5=Sell
        MetricSpec("Insider Ownership", True, 0.4),
        MetricSpec("Insider Transactions", True, 0.6),
        MetricSpec("Institutional Ownership", True, 0.4),
        MetricSpec("Institutional Transactions", True, 0.6),
        MetricSpec("Short Float", False, 0.8),
        MetricSpec("Short Ratio", False, 0.6),
        MetricSpec("Relative Volume", True, 0.4),
    ],
}

# Final weights (sum=1.0) emphasizing regularity via Stability
PILLAR_WEIGHTS = {
    "Valuation": 0.18,
    "Quality":   0.22,
    "Growth":    0.18,
    "Momentum":  0.16,
    "Stability": 0.20,
    "Sentiment": 0.06,
}


PERF_COLS = [
    "Performance (Week)",
    "Performance (Month)",
    "Performance (Quarter)",
    "Performance (Half Year)",
    "Performance (YTD)",
    "Performance (Year)",
    "Performance (3 Years)",
    "Performance (5 Years)",
    "Performance (10 Years)",
]


def compute_regularity_penalty(df_num: pd.DataFrame, group: Optional[pd.Series]) -> pd.Series:
    """Compute an 'irregularity' penalty (0..20-ish points) based on dispersion of returns and volatility.
    Lower is better. Penalty is applied to FinalScore.
    """
    cols = [c for c in PERF_COLS if c in df_num.columns]
    if not cols:
        return pd.Series(np.nan, index=df_num.index)

    perf = df_num[cols]
    # Dispersion across horizons: std of available performance metrics
    disp = perf.std(axis=1, skipna=True)

    # Add explicit vol components if present
    vol_parts = []
    if "Volatility (Week)" in df_num.columns:
        vol_parts.append(df_num["Volatility (Week)"])
    if "Volatility (Month)" in df_num.columns:
        vol_parts.append(df_num["Volatility (Month)"])
    if "Beta" in df_num.columns:
        vol_parts.append(df_num["Beta"])

    if vol_parts:
        vol_stack = pd.concat(vol_parts, axis=1)
        vol_index = vol_stack.mean(axis=1, skipna=True)
        raw = disp.fillna(0) + 0.5 * vol_index.fillna(0)
    else:
        raw = disp

    raw = winsorize_series(raw, 0.01, 0.99)

    # Convert to percentile where lower irregularity = better -> higher score
    reg_score = pct_rank(raw, higher_is_better=False, group=group)

    # Penalty: map low regularity to higher penalty; max ~20 points
    penalty = (100.0 - reg_score) * 0.20
    return penalty




def compute_growth_penalty(growth_score: pd.Series, scale: float = 0.20) -> pd.Series:
    """Penalty (0..~20 points) derived from Growth Score (0..100).
    Higher Growth Score => lower penalty. Applied to FinalScore.
    """
    if growth_score is None:
        return pd.Series(np.nan)
    return (100.0 - growth_score) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="finviz.csv", help="Input Finviz CSV path")
    ap.add_argument("--output", default="scored.csv", help="Output scored CSV path")
    ap.add_argument("--sector_neutral", action="store_true", help="Percentiles within Sector (recommended)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Basic identifiers
    id_cols = [c for c in ["Ticker", "Company", "Sector", "Industry", "Country", "Exchange", "Asset Type"] if c in df.columns]

    # Parse numeric only for columns that feed scoring (fast + scalable).
    # (We still keep the original df for text fields, dates, URLs, etc.)
    needed_cols = set(PERF_COLS)
    for specs in PILLARS.values():
        for m in specs:
            needed_cols.add(m.col)
    # Common useful extras (won't break if missing)
    needed_cols |= {"Price", "Market Cap", "Average Volume", "Volume", "Target Price",
                    "Assets Under Management", "Net Expense Ratio",
                    "Net Flows (1 Month)", "Net Flows % (1 Month)",
                    "Net Flows (3 Month)", "Net Flows % (3 Month)",
                    "Net Flows (YTD)", "Net Flows % (YTD)",
                    "Return 1 Year", "Return 3 Year", "Return 5 Year",
                    "Employees", "Shares Outstanding", "Shares Float"}
    needed_cols = [c for c in needed_cols if c in df.columns and c not in id_cols]

    numeric_data = {col: df[col].map(parse_numeric) for col in needed_cols}
    df_num = pd.DataFrame(numeric_data, index=df.index)

    # Optional sector-neutral scoring
    group = df["Sector"] if (args.sector_neutral and "Sector" in df.columns) else None

    # Winsorize numeric inputs for metric columns to stabilize percentiles
    # (Only for columns that are used by specs or for penalty.)
    used_cols = set(PERF_COLS)
    for specs in PILLARS.values():
        for m in specs:
            used_cols.add(m.col)
    used_cols = [c for c in used_cols if c in df_num.columns]
    for c in used_cols:
        df_num[c] = winsorize_series(df_num[c], 0.01, 0.99)

    # Special transform: RSI (14) -> stability-friendly score before percentile
    if "Relative Strength Index (14)" in df_num.columns:
        df_num["_RSI_SCORE"] = rsi_score_linear(df_num["Relative Strength Index (14)"])
    else:
        df_num["_RSI_SCORE"] = np.nan

    # Keep ALL original columns
    out = df.copy()

    # Compute pillar scores
    pillar_scores: Dict[str, pd.Series] = {}
    pillar_coverage: Dict[str, pd.Series] = {}

    for pillar, specs in PILLARS.items():
        values = []
        weights = []
        used = []

        for m in specs:
            col = m.col
            if col not in df_num.columns:
                continue

            series = df_num[col].copy()

            # Swap in RSI transform
            if col == "Relative Strength Index (14)":
                series = df_num["_RSI_SCORE"]

            score = pct_rank(series, higher_is_better=m.higher_is_better, group=group)
            values.append(score)
            weights.append(m.weight)
            used.append(col)

        if not values:
            pillar_scores[pillar] = pd.Series(np.nan, index=df.index)
            pillar_coverage[pillar] = pd.Series(0, index=df.index)
            continue

        pillar_scores[pillar] = safe_weighted_mean(values, weights)
        # coverage: number of non-NaN metrics used for this row
        mat = pd.concat(values, axis=1)
        pillar_coverage[pillar] = mat.notna().sum(axis=1)

        out[f"{pillar} Score"] = pillar_scores[pillar].round(2)
        out[f"{pillar} Metrics Used"] = pillar_coverage[pillar].astype(int)

    # Regularity penalty (emphasize smoothness)
    penalty = compute_regularity_penalty(df_num, group=group)
    out["Irregularity Penalty"] = penalty.round(2)

    # Growth penalty (reward higher Growth Score; penalize low growth)
    growth_penalty = compute_growth_penalty(pillar_scores.get("Growth"), scale=0.20)
    out["Growth Penalty"] = growth_penalty.round(2)

    # Final score (weighted mean of pillar scores) - penalty
    # If some pillars NaN, renormalize weights by availability.
    pillars = list(PILLAR_WEIGHTS.keys())
    score_mat = np.vstack([pillar_scores[p].to_numpy(dtype=float) for p in pillars]).T
    w = np.array([PILLAR_WEIGHTS[p] for p in pillars], dtype=float)
    mask = ~np.isnan(score_mat)
    w_mat = mask * w
    denom = w_mat.sum(axis=1)
    numer = np.nansum(score_mat * w, axis=1)
    base = np.where(denom > 0, numer / denom, np.nan)

    final = base - penalty.to_numpy(dtype=float) - growth_penalty.to_numpy(dtype=float)
    out["Base Score"] = np.round(base, 2)
    out["Final Score"] = np.round(final, 2)

    # Rank (descending)
    out["Rank"] = out["Final Score"].rank(ascending=False, method="min")
    out = out.sort_values("Final Score", ascending=False)



    # Reorder columns: Final Score first, then ALL original input columns (unchanged order),
    # then scoring diagnostics to the far right.
    original_cols = list(df.columns)
    diag_cols = [
        "Rank",
        "Base Score",
        "Irregularity Penalty",
        "Growth Penalty",
    ]
    for p in PILLAR_WEIGHTS.keys():
        diag_cols += [f"{p} Score", f"{p} Metrics Used"]

    ordered = ["Final Score"]
    for c in original_cols:
        if c in out.columns and c not in ordered:
            ordered.append(c)
    for c in diag_cols:
        if c in out.columns and c not in ordered:
            ordered.append(c)
    # Any remaining new columns (future-proof)
    for c in out.columns:
        if c not in ordered:
            ordered.append(c)

    out = out.loc[:, ordered]

    out.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}  (rows={len(out)}, cols={out.shape[1]})")


if __name__ == "__main__":
    main()
