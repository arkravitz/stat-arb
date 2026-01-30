from pathlib import Path

import pandas as pd

from .backtests import BacktestResult


def build_backtest_attribution_summary(result: BacktestResult) -> pd.DataFrame:
    ts = result.timeseries
    rows: list[dict[str, object]] = []

    for column in sorted(col for col in ts.columns if col.startswith("asset_contrib_")):
        rows.append(
            {
                "category": "asset",
                "name": column.removeprefix("asset_contrib_"),
                "total": float(ts[column].sum()),
            }
        )
    for column in sorted(
        col for col in ts.columns if col.startswith("factor_contrib_")
    ):
        rows.append(
            {
                "category": "factor",
                "name": column.removeprefix("factor_contrib_"),
                "total": float(ts[column].sum()),
            }
        )
    for column in ["transaction_cost", "borrow_cost"]:
        if column in ts.columns:
            rows.append(
                {"category": "cost", "name": column, "total": float(-ts[column].sum())}
            )

    if "event_blackout" in ts.columns:
        rows.append(
            {
                "category": "constraint",
                "name": "event_blackout_days",
                "total": int(ts["event_blackout"].sum()),
            }
        )
    if "regime_ok" in ts.columns:
        blocked = int((~ts["regime_ok"].fillna(False)).sum())
        rows.append(
            {"category": "constraint", "name": "regime_blocked_days", "total": blocked}
        )
    if "capital_used" in ts.columns:
        rows.append(
            {
                "category": "portfolio",
                "name": "avg_capital_used",
                "total": float(ts["capital_used"].mean()),
            }
        )

    rows.append(
        {
            "category": "portfolio",
            "name": "net_total_return",
            "total": float(result.stats["total_return"]),
        }
    )
    return pd.DataFrame(rows)


def build_walk_forward_selection_attribution(
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    if fold_summary.empty or "trade" not in fold_summary.columns:
        return pd.DataFrame(
            columns=["trade", "selected_folds", "avg_weight", "avg_test_sharpe"]
        )

    selected = fold_summary[fold_summary["rank"] > 0].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=["trade", "selected_folds", "avg_weight", "avg_test_sharpe"]
        )
    if "archetype" not in selected.columns:
        selected["archetype"] = "unknown"

    grouped = (
        selected.groupby("trade", as_index=False)
        .agg(
            archetype=("archetype", "first"),
            selected_folds=("fold", "count"),
            avg_weight=("portfolio_weight", "mean"),
            avg_test_sharpe=("test_sharpe", "mean"),
            avg_selection_score=("selection_score", "mean"),
        )
        .sort_values(
            ["selected_folds", "avg_selection_score"], ascending=[False, False]
        )
    )
    return grouped


def write_backtest_attribution_files(
    result: BacktestResult,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{stem}_summary.csv"
    daily_path = output_dir / f"{stem}_daily.csv"

    summary = build_backtest_attribution_summary(result)
    daily_columns = [
        column
        for column in result.timeseries.columns
        if column.startswith("asset_contrib_")
        or column.startswith("factor_contrib_")
        or column
        in {
            "portfolio_returns",
            "gross_spread_returns",
            "transaction_cost",
            "borrow_cost",
            "position",
        }
    ]
    summary.to_csv(summary_path, index=False)
    result.timeseries[daily_columns].to_csv(daily_path)
    return summary_path, daily_path


def write_walk_forward_attribution_file(
    fold_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_walk_forward_selection_attribution(fold_summary).to_csv(
        output_path, index=False
    )
    return output_path
