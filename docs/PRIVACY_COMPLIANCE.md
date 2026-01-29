# Privacy & Compliance Checklist

## 1) Never commit secrets
- Do **not** commit `.env`, API keys, tokens, private emails, or local paths.
- Use `.env.example` for placeholders and `.gitignore` to exclude local files.

## 2) SEC EDGAR etiquette
- Set `SEC_USER_AGENT` via environment variable (required for responsible automated access).
- Recommended: use a dedicated alias email if you want to keep your personal mailbox private.

Example:
```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

## 3) Finviz handling
- Prefer **manual export** (CSV you already downloaded).
- Avoid publishing large/raw Finviz exports; share only a small anonymized sample if needed.
- Avoid scraping Finviz in public code unless you have explicit permission and ToS clarity.

## 4) Outputs & cache files
- Do not commit:
  - full `*.csv` outputs,
  - caches (`cache_*`),
  - pickles, logs.
- Commit only `data/sample_*.csv` and `outputs/sample_*.csv`.

## 5) Cleaning Git history (if you already pushed something sensitive)
- Use `git filter-repo` to remove secrets from history, then force-push.
- Rotate any leaked keys immediately.

## 6) “Public but safe” defaults
- Replace any hard-coded personal email/user-agent in source code with placeholders.
- Keep verbose logs off by default (avoid printing full file paths, home dirs, etc.).

