# Repository Guidelines

## Project Structure & Module Organization
This repository is a small research codebase built around three top-level Python scripts. `stat_arb_bucket_finder.py` generates candidate baskets and writes `stat_arb_buckets.csv`, `stat_arb_buckets.json`, and `stat_arb_buckets_llm_prompt.txt`. `basket_backtester.py` backtests a single basket from cached prices. `basket_backtester_batch.py` runs batch backtests from a CSV `trade` column. Cached market data lives in `price_cache/`; generated reports, charts, and timeseries belong in `outputs/`.

## Build, Test, and Development Commands
Dependencies are declared in `pyproject.toml`; use `uv sync --all-extras` to create the environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib scipy scikit-learn statsmodels torch yfinance requests tqdm pyarrow
python stat_arb_bucket_finder.py --universe-preset semis_fast --provider yfinance
python basket_backtester.py --basket "long 1 units of AVGO, short 5 units of SMH"
python basket_backtester_batch.py --csv stat_arb_buckets.csv --output-dir outputs/all_buckets
python -m py_compile stat_arb_bucket_finder.py basket_backtester.py basket_backtester_batch.py
```

Use the final command as a lightweight syntax check before opening a PR.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions and variables, PascalCase for dataclasses and model classes, and uppercase module constants such as `DEFAULT_CACHE_DIRS`. Keep scripts directly runnable from the repository root. Prefer small, explicit helper functions over dense inline logic. When adding CLI flags, use descriptive long options such as `--force-refresh` or `--output-dir`.

## Testing Guidelines
Formal tests live in `tests/` using `test_<module>.py` naming; run `uv run pytest -q` before opening a PR, and include a reproducible smoke command in the PR description.

## Commit & Pull Request Guidelines
Git history is minimal (`initial commit`), so use short imperative subjects going forward, for example: `Add parquet cache fallback` or `Tighten basket parser validation`. PRs should include the purpose of the change, exact commands run, any new data or output paths touched, and screenshots only when chart output changes materially.

## Data & Configuration Notes
Do not commit API keys. Pass vendor credentials through `--api-key` or environment variables, and keep large generated artifacts in `outputs/` and `price_cache/` out of review unless they are necessary to explain a result.
