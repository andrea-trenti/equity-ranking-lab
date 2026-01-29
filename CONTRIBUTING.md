# Contributing

## Scope
PRs are welcome for:
- Better parsing/robustness
- Clearer scoring explanations and tests
- Faster caching and safer retries/backoff
- Adding CLI flags (argparse) without breaking defaults

## Style
- Python 3.10+
- Type hints where practical
- Keep outputs deterministic (seed randomness if used)

## Testing
If you add tests, keep them fast and network-free (mock network calls).

