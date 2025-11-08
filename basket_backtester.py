import argparse
from pathlib import Path

import pandas as pd

from stat_arb.backtests import PortfolioConfig, run_weighted_portfolio_backtest
from stat_arb.baskets import parse_basket_string, parse_weights_json, render_weight_lines
from stat_arb.prices import load_price_panel, resolve_cache_dirs
from stat_arb.reports import write_backtest_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a custom long/short basket using cached price data."
    )
    parser.add_argument(
        "--basket",
        type=str,
        help="Basket string like 'long 1 units of AVGO, short 5 units of SMH'",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help='Python-dict style weights like "{\'AVGO\': 1, \'SMH\': -5}"',
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional explicit cache directory")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--rebalance",
        choices=["daily", "weekly", "monthly", "none"],
        default="daily",
    )
    parser.add_argument("--report-path", type=str, default="outputs/basket_backtest.html")
    parser.add_argument("--plot-path", dest="report_path", help=argparse.SUPPRESS)
    parser.add_argument("--stats-path", type=str, default="outputs/basket_stats.csv")
    parser.add_argument("--series-path", type=str, default="outputs/basket_timeseries.csv")
    return parser.parse_args()


def _filter_prices(prices: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        prices = prices.loc[pd.to_datetime(start):]
    if end:
        prices = prices.loc[: pd.to_datetime(end)]
    prices = prices.dropna()
    if prices.empty:
        raise RuntimeError("No overlapping price history found after filtering dates and dropping NaNs.")
    return prices


def main() -> None:
    args = parse_args()
    if not args.basket and not args.weights:
        raise SystemExit("Provide either --basket or --weights")

    spec = parse_basket_string(args.basket) if args.basket else parse_weights_json(args.weights)
    print("Basket weights:")
    for line in render_weight_lines(spec.weights):
        print(line)

    cache_dirs = resolve_cache_dirs(args.cache_dir)
    print(f"Using cache dirs: {[str(path) for path in cache_dirs]}")

    prices = load_price_panel(list(spec.weights.keys()), cache_dirs)
    prices = _filter_prices(prices, args.start, args.end)

    result = run_weighted_portfolio_backtest(
        prices=prices,
        weights=spec.weights,
        config=PortfolioConfig(rebalance=args.rebalance),
        basket_label=args.basket or args.weights,
    )

    series_path = Path(args.series_path)
    stats_path = Path(args.stats_path)
    report_path = Path(args.report_path)
    series_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    result.timeseries.to_csv(series_path)
    result.stats.to_frame("value").to_csv(stats_path)
    write_backtest_report(result, report_path, title="Basket Backtest")

    print("\nBacktest stats:")
    print(result.stats)
    print(f"\nSaved time series to {series_path}")
    print(f"Saved stats to {stats_path}")
    print(f"Saved interactive report to {report_path}")


if __name__ == "__main__":
    main()
