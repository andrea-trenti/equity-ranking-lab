# Usage Examples

## A) Yahoo Finance (lite)
```bash
python script_lite.py
```
Produces: `sp500_multifactor_ranking_lite.csv`

## B) Yahoo + SEC (full)
```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
python script3.py
```
Produces:
- `sp500_multifactor_ranking.csv`
- `coverage_report.csv`

## C) Finviz CSV scoring
```bash
python finviz_scoring.py --input finviz.csv --output scored_finviz.csv --sector_neutral
```

## Recommended repo layout
```
.
├── docs/
├── data/              # only sample_*.csv
├── outputs/           # only sample_*.csv
├── script_lite.py
├── script3.py
├── finviz_scoring.py
├── requirements.txt
├── .gitignore
└── .env.example
```

