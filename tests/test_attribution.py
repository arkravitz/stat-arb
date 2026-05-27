from pathlib import Path

import pandas as pd

from stat_arb.attribution import (
    build_backtest_attribution_summary,
    build_walk_forward_selection_attribution,
    write_backtest_attribution_files,
)
from stat_arb.backtests import PortfolioConfig, run_weighted_portfolio_backtest


def test_build_backtest_attribution_summary_and_files(tmp_path: Path) -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104],
            "BBB": [50, 49, 50, 51, 52],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    result = run_weighted_portfolio_backtest(
        prices=prices,
        weights={"AAA": 1.0, "BBB": -1.0},
        config=PortfolioConfig(rebalance="daily"),
        basket_label="attrib basket",
    )

    summary = build_backtest_attribution_summary(result)
    summary_path, daily_path = write_backtest_attribution_files(
        result, tmp_path, "attrib"
    )

    assert {"category", "name", "total"}.issubset(summary.columns)
    assert summary_path.exists()
    assert daily_path.exists()


def test_build_walk_forward_selection_attribution() -> None:
    fold_summary = pd.DataFrame(
        [
            {
                "fold": 1,
                "rank": 1,
                "trade": "t1",
                "portfolio_weight": 0.6,
                "test_sharpe": 1.2,
                "selection_score": 0.8,
            },
            {
                "fold": 1,
                "rank": 2,
                "trade": "t2",
                "portfolio_weight": 0.4,
                "test_sharpe": 0.7,
                "selection_score": 0.5,
            },
            {
                "fold": 2,
                "rank": 1,
                "trade": "t1",
                "portfolio_weight": 1.0,
                "test_sharpe": 0.9,
                "selection_score": 0.6,
            },
        ]
    )

    summary = build_walk_forward_selection_attribution(fold_summary)

    assert list(summary["trade"]) == ["t1", "t2"]
    assert float(summary.iloc[0]["selected_folds"]) == 2
