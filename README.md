# S&P 500 Multifactor Ranking + Finviz CSV Scoring

This repository contains **3** reproducible pipelines for scoring and ranking public equities:
1) **Yahoo Finance (lite)**: high-coverage multifactor ranking from prices + fast Yahoo snapshot (`script_lite.py`).
2) **Yahoo + SEC (full)**: expanded multifactor ranking with official SEC EDGAR “companyfacts” history (`script3.py`).
3) **Finviz CSV scoring**: score a user-provided Finviz export (`finviz_scoring.py`).

> **Not investment advice.** This is a research/education project.

---

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Run Yahoo Finance (lite)
```bash
python script_lite.py
```
Output: `sp500_multifactor_ranking_lite.csv`

### 3) Run Yahoo + SEC (full)
**Important:** set an EDGAR User-Agent (see `docs/PRIVACY_COMPLIANCE.md`).
```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
python script3.py
```
Outputs: `sp500_multifactor_ranking.csv`, `coverage_report.csv`

### 4) Run Finviz CSV scoring
```bash
python finviz_scoring.py --input finviz.csv --output scored_finviz.csv --sector_neutral
```

---

## Files

- `script_lite.py` — S&P 500 “lite” ranking (prices + Yahoo snapshot; retries/backoff).
- `script3.py` — S&P 500 “full” ranking (adds SEC history; optional forward estimates).
- `finviz_scoring.py` — Finviz-style CSV → scored CSV (6 pillar scores + final score).

See:
- `docs/USAGE_EXAMPLES.md`
- `docs/METHODOLOGY.md`
- `docs/PRIVACY_COMPLIANCE.md`
- `docs/DATA_SOURCES_LICENSES.md`

---

## Outputs (high level)

- `Rank`, `Rating`, `TotalScore` (and adjusted score if present)
- Pillar/category scores (where implemented)
- Coverage columns (how many metrics were available per ticker)
- Raw metrics to the right (so you can audit the scoring)

---

## Repository hygiene

- Commit only **sample** inputs/outputs (small subsets).  
- Never commit `.env`, API keys, caches, or full-size CSV outputs. See `.gitignore`.

---

## License

MIT (see `LICENSE`). Replace placeholders with your preferred name/handle if you want attribution.

