# stat-arb

Research utilities for discovering and backtesting statistical-arbitrage baskets in equity markets.

The pipeline has three moving parts: a bucket finder that scores candidate stock buckets from a universe export, a basket optimizer that sizes long/short legs inside each bucket, and a batch backtester that runs walk-forward evaluations across the whole universe. Results land in `outputs/`, which stays local.

## Layout

```
stat_arb/                    library package
  prices.py                  price ingestion and alignment
  events.py                  event-study primitives
  baskets.py                 basket construction
  factors.py                 factor exposures
  structure.py               bucket structure analysis
  optimizer.py               long/short sizing inside a bucket
  backtests.py               backtest engine
  walkforward.py             walk-forward harness
  attribution.py             pnl attribution
  metrics.py                 performance metrics
  reports.py                 run reports
stat_arb_bucket_finder.py    CLI: mine the bucket universe
basket_backtester.py         CLI: backtest a single basket
basket_backtester_batch.py   CLI: sweep the universe
stat_arb_buckets.csv/.json   current bucket snapshot (checked in for reproducibility)
tests/                       unit tests per module
```

## Quickstart

```bash
uv sync --all-extras
uv run pytest -q
uv run python stat_arb_bucket_finder.py --help
```

The finder and backtester CLIs read provider keys from environment variables; `--help` lists the exact names. A sample bucket prompt used for LLM-assisted bucket research is checked in as `stat_arb_buckets_llm_prompt.txt`.

## Data

`price_cache/` and `outputs/` are local artifacts and gitignored. The bucket snapshot files are small and checked in so runs are reproducible.

## Tests

```bash
uv run pytest -q
```

22 tests cover prices, events, baskets, factors, the optimizer, backtests, walk-forward, attribution, reports, and the bucket finder.

## License

MIT.
