from pathlib import Path

import pandas as pd

from stat_arb.backtests import (
    MeanReversionConfig,
    PortfolioConfig,
    run_mean_reversion_backtest,
    run_weighted_portfolio_backtest,
)
from stat_arb.reports import write_backtest_report, write_batch_dashboard


def _sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 103],
            "BBB": [50, 49, 51, 52, 51, 50],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )


def test_write_weighted_backtest_report(tmp_path: Path) -> None:
    result = run_weighted_portfolio_backtest(
        prices=_sample_prices(),
        weights={"AAA": 1.0, "BBB": -1.0},
        config=PortfolioConfig(rebalance="daily"),
        basket_label="weighted basket",
    )
    output_path = tmp_path / "weighted_report.html"

    write_backtest_report(result, output_path, title="Weighted Report")

    text = output_path.read_text(encoding="utf-8")
    assert "Weighted Report" in text
    assert "plotly" in text.lower()
    assert "weighted basket" in text


def test_write_mean_reversion_report_and_dashboard(tmp_path: Path) -> None:
    result = run_mean_reversion_backtest(
        prices=_sample_prices(),
        weights={"AAA": 1.0, "BBB": -1.0},
        config=MeanReversionConfig(entry_z=0.5, exit_z=0.0, lookback=2),
        basket_label="mean reversion basket",
    )
    report_path = tmp_path / "mean_reversion_report.html"
    dashboard_path = tmp_path / "dashboard.html"

    write_backtest_report(
        result,
        report_path,
        title="Mean Reversion Report",
        config=MeanReversionConfig(entry_z=0.5, exit_z=0.0, lookback=2),
    )
    summary = pd.DataFrame(
        [
            {
                "trade": "long 1 units of AAA, short 1 units of BBB",
                "annual_return": 0.12,
                "annual_vol": 0.2,
                "sharpe": 0.6,
                "max_drawdown": -0.08,
                "entries": 2,
                "active_fraction": 0.4,
                "report_path": str(report_path),
            }
        ]
    )
    write_batch_dashboard(summary, dashboard_path)

    assert report_path.exists()
    assert dashboard_path.exists()
    assert "Mean Reversion Report" in report_path.read_text(encoding="utf-8")
    assert "Batch Backtest Dashboard" in dashboard_path.read_text(encoding="utf-8")
