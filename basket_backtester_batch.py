import argparse
from pathlib import Path

import pandas as pd

from stat_arb.attribution import (
    write_backtest_attribution_files,
    write_walk_forward_attribution_file,
)
from stat_arb.backtests import (
    MeanReversionConfig,
    ResidualStrategyConfig,
    run_mean_reversion_backtest,
    run_residual_backtest,
    run_residual_momentum_backtest,
)
from stat_arb.baskets import parse_basket_string, slugify_trade
from stat_arb.events import (
    build_catalyst_signal,
    build_event_blackout_mask,
    load_event_calendar,
)
from stat_arb.factors import FACTOR_PRESETS, resolve_factor_symbols
from stat_arb.prices import load_price_panel, resolve_cache_dirs
from stat_arb.reports import (
    write_backtest_report,
    write_batch_dashboard,
    write_walk_forward_report,
)
from stat_arb.structure import trade_archetype
from stat_arb.walkforward import (
    CandidateBasket,
    WalkForwardConfig,
    load_candidate_baskets,
    run_walk_forward_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest one basket or evaluate a CSV of candidate baskets."
    )
    parser.add_argument(
        "--basket",
        type=str,
        help="Basket string like 'long 1 units of AVGO, short 5 units of SMH'",
    )
    parser.add_argument("--csv", type=str, help="CSV file containing candidate baskets")
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Optional explicit cache directory"
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--strategy",
        choices=["mean_reversion", "residual", "residual_momentum"],
        default="mean_reversion",
        help="Signal engine to use for trading candidate baskets",
    )
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.0)
    parser.add_argument(
        "--lookback", type=int, default=60, help="Signal lookback window"
    )
    parser.add_argument(
        "--factor-symbols",
        nargs="+",
        default=["SPY", "QQQ", "XLK", "SMH", "SOXX"],
        help="Factor hedge symbols for the residual strategy",
    )
    parser.add_argument("--factor-preset", choices=sorted(FACTOR_PRESETS), default=None)
    parser.add_argument("--beta-lookback", type=int, default=126)
    parser.add_argument(
        "--hedge-rebalance",
        choices=["daily", "weekly", "monthly"],
        default="weekly",
    )
    parser.add_argument("--target-vol", type=float, default=0.12)
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--max-gross-leverage", type=float, default=1.5)
    parser.add_argument("--max-residual-vol", type=float, default=0.35)
    parser.add_argument("--beta-stability-lookback", type=int, default=20)
    parser.add_argument("--max-beta-instability", type=float, default=0.15)
    parser.add_argument("--max-factor-vol", type=float, default=0.35)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--borrow-cost-bps", type=float, default=30.0)
    parser.add_argument("--events-csv", type=str, default=None)
    parser.add_argument("--blackout-days-before", type=int, default=2)
    parser.add_argument("--blackout-days-after", type=int, default=1)
    parser.add_argument("--catalyst-days-after", type=int, default=20)
    parser.add_argument("--require-catalyst", action="store_true")
    parser.add_argument("--compare-residual-modes", action="store_true")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Select baskets on train/validation windows and evaluate only the following test window",
    )
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--validate-days", type=int, default=126)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-train-sharpe", type=float, default=0.25)
    parser.add_argument("--min-validation-sharpe", type=float, default=0.5)
    parser.add_argument("--min-validation-entries", type=int, default=3)
    parser.add_argument("--stability-penalty", type=float, default=0.5)
    parser.add_argument("--max-symbol-overlap", type=float, default=0.5)
    parser.add_argument("--max-validation-correlation", type=float, default=0.8)
    return parser.parse_args()


def _filter_prices(
    prices: pd.DataFrame, start: str | None, end: str | None
) -> pd.DataFrame:
    if start:
        prices = prices.loc[pd.to_datetime(start) :]
    if end:
        prices = prices.loc[: pd.to_datetime(end)]
    prices = prices.dropna()
    if prices.empty:
        raise RuntimeError(
            "No overlapping price history found after filtering dates and dropping NaNs."
        )
    return prices


def _candidate_from_basket_text(basket_text: str) -> CandidateBasket:
    spec = parse_basket_string(basket_text)
    return CandidateBasket(
        trade=basket_text, weights=spec.weights, source_row={"trade": basket_text}
    )


def _load_master_panel(
    candidates: list[CandidateBasket],
    cache_dirs: list[Path],
    start: str | None,
    end: str | None,
    factor_symbols: list[str],
) -> pd.DataFrame:
    symbols = sorted(
        set(symbol for candidate in candidates for symbol in candidate.weights).union(
            factor_symbols
        )
    )
    prices = load_price_panel(symbols, cache_dirs)
    return _filter_prices(prices, start, end)


def _build_result_from_panel(
    panel: pd.DataFrame,
    candidate: CandidateBasket,
    strategy: str,
    mean_reversion_config: MeanReversionConfig,
    residual_config: ResidualStrategyConfig,
    event_calendar: pd.DataFrame | None,
    blackout_days_before: int,
    blackout_days_after: int,
    catalyst_days_after: int,
    require_catalyst: bool,
):
    asset_prices = panel[list(candidate.weights.keys())]
    blackout_mask = build_event_blackout_mask(
        asset_prices.index,
        list(candidate.weights.keys()),
        event_calendar,
        days_before=blackout_days_before,
        days_after=blackout_days_after,
    )
    catalyst_signal = build_catalyst_signal(
        asset_prices.index,
        list(candidate.weights.keys()),
        event_calendar,
        days_after=catalyst_days_after,
    )
    if strategy == "mean_reversion":
        return run_mean_reversion_backtest(
            prices=asset_prices,
            weights=candidate.weights,
            config=mean_reversion_config,
            basket_label=candidate.trade,
            blackout_mask=blackout_mask,
        )
    factor_prices = panel[list(residual_config.factor_symbols)]
    if strategy == "residual_momentum":
        return run_residual_momentum_backtest(
            asset_prices=asset_prices,
            weights=candidate.weights,
            factor_prices=factor_prices,
            config=residual_config,
            basket_label=candidate.trade,
            blackout_mask=blackout_mask,
            catalyst_signal=catalyst_signal,
            require_catalyst=require_catalyst,
        )
    return run_residual_backtest(
        asset_prices=asset_prices,
        weights=candidate.weights,
        factor_prices=factor_prices,
        config=residual_config,
        basket_label=candidate.trade,
        blackout_mask=blackout_mask,
        catalyst_signal=catalyst_signal,
        require_catalyst=require_catalyst,
    )


def run_batch_from_candidates(
    candidates: list[CandidateBasket],
    panel: pd.DataFrame,
    output_dir: Path,
    strategy: str,
    mean_reversion_config: MeanReversionConfig,
    residual_config: ResidualStrategyConfig,
    event_calendar: pd.DataFrame | None,
    blackout_days_before: int,
    blackout_days_after: int,
    catalyst_days_after: int,
    require_catalyst: bool,
) -> pd.DataFrame:
    reports_dir = output_dir / "reports"
    series_dir = output_dir / "timeseries"
    attribution_dir = output_dir / "attribution"
    reports_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)
    attribution_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for idx, candidate in enumerate(candidates, start=1):
        result = _build_result_from_panel(
            panel=panel,
            candidate=candidate,
            strategy=strategy,
            mean_reversion_config=mean_reversion_config,
            residual_config=residual_config,
            event_calendar=event_calendar,
            blackout_days_before=blackout_days_before,
            blackout_days_after=blackout_days_after,
            catalyst_days_after=catalyst_days_after,
            require_catalyst=require_catalyst,
        )

        slug = slugify_trade(candidate.trade)
        report_path = reports_dir / f"{idx:02d}_{slug}.html"
        series_path = series_dir / f"{idx:02d}_{slug}.csv"
        attribution_summary_path, attribution_daily_path = (
            write_backtest_attribution_files(
                result,
                attribution_dir,
                f"{idx:02d}_{slug}",
            )
        )

        write_backtest_report(
            result,
            report_path,
            title=f"{strategy.replace('_', ' ').title()} Backtest #{idx}",
            config=mean_reversion_config if strategy == "mean_reversion" else None,
        )
        result.timeseries.to_csv(series_path)

        combined = dict(candidate.source_row)
        combined.setdefault("trade", candidate.trade)
        combined["weights"] = str(candidate.weights)
        combined.update(result.stats.to_dict())
        combined["strategy"] = result.strategy
        combined["archetype"] = trade_archetype(candidate.weights.keys())
        combined["report_path"] = str(report_path)
        combined["timeseries_path"] = str(series_path)
        combined["attribution_summary_path"] = str(attribution_summary_path)
        combined["attribution_daily_path"] = str(attribution_daily_path)
        all_rows.append(combined)

    if not all_rows:
        raise RuntimeError("No trades were processed from the candidate set.")

    summary = pd.DataFrame(all_rows)
    summary_path = output_dir / "all_bucket_backtest_stats.csv"
    summary.to_csv(summary_path, index=False)
    write_batch_dashboard(summary, output_dir / "batch_backtest_dashboard.html")
    return summary


def run_single_candidate(
    candidate: CandidateBasket,
    panel: pd.DataFrame,
    output_dir: Path,
    strategy: str,
    mean_reversion_config: MeanReversionConfig,
    residual_config: ResidualStrategyConfig,
    event_calendar: pd.DataFrame | None,
    blackout_days_before: int,
    blackout_days_after: int,
    catalyst_days_after: int,
    require_catalyst: bool,
) -> None:
    result = _build_result_from_panel(
        panel=panel,
        candidate=candidate,
        strategy=strategy,
        mean_reversion_config=mean_reversion_config,
        residual_config=residual_config,
        event_calendar=event_calendar,
        blackout_days_before=blackout_days_before,
        blackout_days_after=blackout_days_after,
        catalyst_days_after=catalyst_days_after,
        require_catalyst=require_catalyst,
    )
    timeseries_path = output_dir / "basket_timeseries.csv"
    stats_path = output_dir / "basket_stats.csv"
    report_path = output_dir / "basket_backtest.html"
    attribution_dir = output_dir / "attribution"

    result.timeseries.to_csv(timeseries_path)
    result.stats.to_frame("value").to_csv(stats_path)
    write_backtest_report(
        result,
        report_path,
        title=f"{strategy.replace('_', ' ').title()} Basket Backtest",
        config=mean_reversion_config if strategy == "mean_reversion" else None,
    )
    attribution_summary_path, attribution_daily_path = write_backtest_attribution_files(
        result,
        attribution_dir,
        "basket_backtest",
    )

    print("\nBacktest stats:")
    print(result.stats)
    print(f"\nSaved time series to {timeseries_path}")
    print(f"Saved stats to {stats_path}")
    print(f"Saved interactive report to {report_path}")
    print(f"Saved attribution summary to {attribution_summary_path}")
    print(f"Saved attribution daily file to {attribution_daily_path}")


