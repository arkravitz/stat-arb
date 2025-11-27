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


def run_walk_forward(
    candidates: list[CandidateBasket],
    panel: pd.DataFrame,
    output_dir: Path,
    strategy: str,
    mean_reversion_config: MeanReversionConfig,
    residual_config: ResidualStrategyConfig,
    walk_forward_config: WalkForwardConfig,
    event_calendar: pd.DataFrame | None,
    blackout_days_before: int,
    blackout_days_after: int,
    catalyst_days_after: int,
    require_catalyst: bool,
) -> None:
    result = _compute_walk_forward_result(
        candidates=candidates,
        panel=panel,
        strategy=strategy,
        mean_reversion_config=mean_reversion_config,
        residual_config=residual_config,
        walk_forward_config=walk_forward_config,
        event_calendar=event_calendar,
        blackout_days_before=blackout_days_before,
        blackout_days_after=blackout_days_after,
        catalyst_days_after=catalyst_days_after,
        require_catalyst=require_catalyst,
    )

    timeseries_path = output_dir / "walk_forward_timeseries.csv"
    stats_path = output_dir / "walk_forward_stats.csv"
    folds_path = output_dir / "walk_forward_folds.csv"
    report_path = output_dir / "walk_forward_report.html"
    attribution_path = output_dir / "walk_forward_selection_attribution.csv"

    result.timeseries.to_csv(timeseries_path)
    result.stats.to_frame("value").to_csv(stats_path)
    result.fold_summary.to_csv(folds_path, index=False)
    write_walk_forward_attribution_file(result.fold_summary, attribution_path)
    write_walk_forward_report(
        result.timeseries,
        result.fold_summary,
        result.stats,
        report_path,
        title=f"Walk-Forward {strategy.replace('_', ' ').title()} Strategy",
    )

    print("\nWalk-forward stats:")
    print(result.stats)
    print(f"\nSaved walk-forward timeseries to {timeseries_path}")
    print(f"Saved walk-forward stats to {stats_path}")
    print(f"Saved walk-forward fold summary to {folds_path}")
    print(f"Saved walk-forward selection attribution to {attribution_path}")
    print(f"Saved walk-forward report to {report_path}")


def _compute_walk_forward_result(
    candidates: list[CandidateBasket],
    panel: pd.DataFrame,
    strategy: str,
    mean_reversion_config: MeanReversionConfig,
    residual_config: ResidualStrategyConfig,
    walk_forward_config: WalkForwardConfig,
    event_calendar: pd.DataFrame | None,
    blackout_days_before: int,
    blackout_days_after: int,
    catalyst_days_after: int,
    require_catalyst: bool,
):
    warmup_bars = (
        max(mean_reversion_config.lookback, 5)
        if strategy == "mean_reversion"
        else max(residual_config.signal_lookback, residual_config.beta_lookback, 5)
    )

    def runner(history: pd.DataFrame, candidate: CandidateBasket):
        return _build_result_from_panel(
            panel=history,
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

    return run_walk_forward_selection(
        prices=panel,
        candidates=candidates,
        config=walk_forward_config,
        runner=runner,
        warmup_bars=warmup_bars,
    )


def run_residual_mode_comparison(
    candidates: list[CandidateBasket],
    panel: pd.DataFrame,
    output_dir: Path,
    residual_config: ResidualStrategyConfig,
    walk_forward_config: WalkForwardConfig,
    event_calendar: pd.DataFrame | None,
    blackout_days_before: int,
    blackout_days_after: int,
    catalyst_days_after: int,
    require_catalyst: bool,
) -> None:
    results = {}
    for strategy in ["residual", "residual_momentum"]:
        strategy_dir = output_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        result = _compute_walk_forward_result(
            candidates=candidates,
            panel=panel,
            strategy=strategy,
            mean_reversion_config=MeanReversionConfig(),
            residual_config=residual_config,
            walk_forward_config=walk_forward_config,
            event_calendar=event_calendar,
            blackout_days_before=blackout_days_before,
            blackout_days_after=blackout_days_after,
            catalyst_days_after=catalyst_days_after,
            require_catalyst=require_catalyst,
        )
        result.timeseries.to_csv(strategy_dir / "walk_forward_timeseries.csv")
        result.stats.to_frame("value").to_csv(strategy_dir / "walk_forward_stats.csv")
        result.fold_summary.to_csv(strategy_dir / "walk_forward_folds.csv", index=False)
        results[strategy] = result.stats

    comparison = pd.DataFrame(results)
    comparison_path = output_dir / "residual_mode_comparison.csv"
    comparison.to_csv(comparison_path)
    print("\nResidual mode comparison:")
    print(comparison)
    print(f"\nSaved residual mode comparison to {comparison_path}")


def main() -> None:
    args = parse_args()
    if not args.basket and not args.csv:
        raise SystemExit("Provide either --basket or --csv")

    cache_dirs = resolve_cache_dirs(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_symbols = resolve_factor_symbols(args.factor_symbols, args.factor_preset)

    mean_reversion_config = MeanReversionConfig(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        lookback=args.lookback,
    )
    residual_config = ResidualStrategyConfig(
        factor_symbols=tuple(factor_symbols),
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        signal_lookback=args.lookback,
        beta_lookback=args.beta_lookback,
        hedge_rebalance=args.hedge_rebalance,
        target_vol=args.target_vol,
        vol_lookback=args.vol_lookback,
        max_gross_leverage=args.max_gross_leverage,
        max_residual_vol=args.max_residual_vol,
        beta_stability_lookback=args.beta_stability_lookback,
        max_beta_instability=args.max_beta_instability,
        max_factor_vol=args.max_factor_vol,
        transaction_cost_bps=args.transaction_cost_bps,
        borrow_cost_bps=args.borrow_cost_bps,
    )
    event_calendar = (
        load_event_calendar(Path(args.events_csv)) if args.events_csv else None
    )

    candidates = (
        load_candidate_baskets(Path(args.csv))
        if args.csv
        else [_candidate_from_basket_text(args.basket)]
    )
    panel = _load_master_panel(
        candidates=candidates,
        cache_dirs=cache_dirs,
        start=args.start,
        end=args.end,
        factor_symbols=factor_symbols
        if args.strategy in {"residual", "residual_momentum"}
        else [],
    )

    if args.compare_residual_modes:
        walk_forward_config = WalkForwardConfig(
            train_days=args.train_days,
            validate_days=args.validate_days,
            test_days=args.test_days,
            top_k=args.top_k,
            min_train_sharpe=args.min_train_sharpe,
            min_validation_sharpe=args.min_validation_sharpe,
            min_validation_entries=args.min_validation_entries,
            stability_penalty=args.stability_penalty,
            max_symbol_overlap=args.max_symbol_overlap,
            max_validation_correlation=args.max_validation_correlation,
        )
        run_residual_mode_comparison(
            candidates=candidates,
            panel=panel,
            output_dir=output_dir,
            residual_config=residual_config,
            walk_forward_config=walk_forward_config,
            event_calendar=event_calendar,
            blackout_days_before=args.blackout_days_before,
            blackout_days_after=args.blackout_days_after,
            catalyst_days_after=args.catalyst_days_after,
            require_catalyst=args.require_catalyst,
        )
        return

    if args.walk_forward:
        walk_forward_config = WalkForwardConfig(
            train_days=args.train_days,
            validate_days=args.validate_days,
            test_days=args.test_days,
            top_k=args.top_k,
            min_train_sharpe=args.min_train_sharpe,
            min_validation_sharpe=args.min_validation_sharpe,
            min_validation_entries=args.min_validation_entries,
            stability_penalty=args.stability_penalty,
            max_symbol_overlap=args.max_symbol_overlap,
            max_validation_correlation=args.max_validation_correlation,
        )
        run_walk_forward(
            candidates=candidates,
            panel=panel,
            output_dir=output_dir,
            strategy=args.strategy,
            mean_reversion_config=mean_reversion_config,
            residual_config=residual_config,
            walk_forward_config=walk_forward_config,
            event_calendar=event_calendar,
            blackout_days_before=args.blackout_days_before,
            blackout_days_after=args.blackout_days_after,
            catalyst_days_after=args.catalyst_days_after,
            require_catalyst=args.require_catalyst,
        )
        return

    if not args.csv:
        run_single_candidate(
            candidate=candidates[0],
            panel=panel,
            output_dir=output_dir,
            strategy=args.strategy,
            mean_reversion_config=mean_reversion_config,
            residual_config=residual_config,
            event_calendar=event_calendar,
            blackout_days_before=args.blackout_days_before,
            blackout_days_after=args.blackout_days_after,
            catalyst_days_after=args.catalyst_days_after,
            require_catalyst=args.require_catalyst,
        )
        return

    summary = run_batch_from_candidates(
        candidates=candidates,
        panel=panel,
        output_dir=output_dir,
        strategy=args.strategy,
        mean_reversion_config=mean_reversion_config,
        residual_config=residual_config,
        event_calendar=event_calendar,
        blackout_days_before=args.blackout_days_before,
        blackout_days_after=args.blackout_days_after,
        catalyst_days_after=args.catalyst_days_after,
        require_catalyst=args.require_catalyst,
    )
    print("\nSaved batch summary to:", output_dir / "all_bucket_backtest_stats.csv")
    print(
        "Saved interactive dashboard to:", output_dir / "batch_backtest_dashboard.html"
    )
    cols = [
        column
        for column in [
            "trade",
            "strategy",
            "annual_return",
            "annual_vol",
            "sharpe",
            "max_drawdown",
            "entries",
            "active_fraction",
            "report_path",
        ]
        if column in summary.columns
    ]
    print(summary[cols])


if __name__ == "__main__":
    main()
