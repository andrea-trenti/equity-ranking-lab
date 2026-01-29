# Data Sources & Licenses (read before public release)

This project **integrates multiple upstream data sources**. You are responsible for verifying that your use complies with their terms.

## Sources used by the scripts
- **S&P 500 constituents list**: pulled from a public list (see README) used only to form the universe.
- **Yahoo Finance data**: accessed via `yfinance` for prices and snapshot fundamentals.
- **SEC EDGAR “companyfacts”**: official XBRL-derived fundamentals history (used in `script3.py`).
- **Finviz CSV**: user-provided export consumed by `finviz_scoring.py`.
- **Optional**: Financial Modeling Prep (FMP) forward estimates if you enable it and provide an API key.

## Practical guidance
- Keep requests modest and cached; avoid high-frequency scraping.
- Do not distribute large raw datasets in Git history; publish only small samples.
- If you publish a paper/blog, cite the upstream providers.

## Licensing notes (non-legal summary)
- Wikipedia content is generally CC BY-SA; SEC materials are U.S. government publications; APIs/market-data providers may have their own terms.
- When in doubt, link to sources and avoid redistributing raw data.

