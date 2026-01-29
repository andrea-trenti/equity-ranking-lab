# Methodology

This project contains two scoring families:

## A) S&P 500 Multifactor Ranking (Yahoo / Yahoo+SEC)

### Data pipeline
- **Universe**: S&P 500 constituents pulled from a public table (see `docs/DATA_SOURCES_LICENSES.md`).
- **Prices**: daily history for momentum, volatility, drawdown, beta (where implemented).
- **Fundamentals**:
  - Lite mode: Yahoo snapshot fields (faster, higher coverage).
  - Full mode: adds official SEC EDGAR “companyfacts” history for longer horizons and robustness.

### Feature engineering (examples)
- Momentum: return windows (e.g., 12–1), multi-year CAGR, and/or recent performance.
- Risk: volatility (annualized), max drawdown, beta vs benchmark (if enabled).
- Fundamentals: growth rates, margins, leverage, valuation ratios (coverage-dependent).

### Scoring
- Metrics are converted to **percentiles** (or a monotone transform) so that:
  - Higher-is-better metrics score up.
  - Lower-is-better (e.g., valuation multiples, leverage) score down.
- If a sector column is present, the scorer can do **sector-neutral** percentiles to reduce structural sector bias.
- **Coverage-aware ranking**: tickers with too few usable metrics are filtered or penalized (so the rank is auditable).

### Outputs
- Ranking columns first (rank/rating/total score), then category scores, then raw metrics to the right.

---

## B) Finviz CSV Scoring (`finviz_scoring.py`)

### Inputs
A Finviz-style CSV export, typically containing:
- Identifiers (Ticker, Company, Sector, …)
- Valuation (P/E, EV/EBITDA, …)
- Quality (ROIC/ROE/margins/leverage, …)
- Growth (EPS & sales growth past/future windows)
- Momentum (returns, performance blocks, RSI, …)
- Stability (volatility/beta, …)
- Sentiment (analyst recomms, short float, insider, …)

### Pillars (6)
- **Valuation**, **Quality**, **Growth**, **Momentum**, **Stability**, **Sentiment**.

### Parsing & normalization
- Robust parsing for percentages, suffixes (K/M/B/T), blanks, and “-”.
- Percentile scoring is applied only where a column exists; missing columns are ignored (graceful degradation).

### Regularity / irregularity penalty
A dedicated penalty is computed to down-rank unstable/irregular profiles (e.g., noisy returns/volatility dispersion).
This penalty is reported explicitly as a column so you can audit it.

---

## Caveats
- Data quality depends on upstream sources; any single-field error can move ranks.
- Sector-neutral scoring is a design choice (trade-off: less sector bias vs potentially masking sector-driven alpha).
- This repo is for research/education; treat outputs as *screeners*, not conclusions.

