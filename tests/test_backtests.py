import pandas as pd

from stat_arb.backtests import (
    MeanReversionConfig,
    PortfolioConfig,
    ResidualStrategyConfig,
    run_mean_reversion_backtest,
    run_residual_backtest,
    run_residual_momentum_backtest,
    run_weighted_portfolio_backtest,
)


def test_weighted_portfolio_backtest_returns_expected_columns() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100, 102, 101, 103, 104],
            "BBB": [50, 49, 50, 51, 52],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )

    result = run_weighted_portfolio_backtest(
        prices=prices,
        weights={"AAA": 1.0, "BBB": -1.0},
        config=PortfolioConfig(rebalance="daily"),
        basket_label="demo basket",
    )

    assert result.strategy == "weighted_portfolio"
    assert {"basket_notional", "portfolio_returns", "equity_curve"}.issubset(
        result.timeseries.columns
    )
    assert result.stats["days"] == 5
    assert result.stats["basket_std"] > 0


def test_mean_reversion_backtest_generates_positions() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100, 100, 100, 100, 100, 100, 100],
            "BBB": [100, 101, 103, 108, 103, 99, 97],
        },
        index=pd.date_range("2024-01-01", periods=7, freq="D"),
    )

    result = run_mean_reversion_backtest(
        prices=prices,
        weights={"AAA": 1.0, "BBB": -1.0},
        config=MeanReversionConfig(entry_z=0.75, exit_z=0.0, lookback=2),
        basket_label="reversion basket",
    )

    assert result.strategy == "mean_reversion"
    assert {"spread_z", "position", "equity_curve"}.issubset(result.timeseries.columns)
    assert result.timeseries["position"].abs().max() == 1.0
    assert result.stats["entries"] >= 1


def test_mean_reversion_blackout_forces_flat_position() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 104],
            "BBB": [100, 99, 98, 97, 96, 95, 96],
        },
        index=pd.date_range("2024-01-01", periods=7, freq="D"),
    )
    blackout = pd.Series(False, index=prices.index)
    blackout.iloc[4:6] = True

    result = run_mean_reversion_backtest(
        prices=prices,
        weights={"AAA": 1.0, "BBB": -1.0},
        config=MeanReversionConfig(entry_z=0.5, exit_z=0.0, lookback=2),
        basket_label="blackout basket",
        blackout_mask=blackout,
    )

    assert (result.timeseries.loc[blackout[blackout].index, "position"] == 0.0).all()


def test_residual_backtest_includes_cost_and_beta_columns() -> None:
    asset_prices = pd.DataFrame(
        {
            "AAA": [100, 101, 103, 102, 101, 100, 102, 103, 101, 100],
            "BBB": [100, 100, 101, 103, 104, 105, 103, 102, 101, 100],
        },
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    factor_prices = pd.DataFrame(
        {
            "QQQ": [100, 100.5, 101, 100.8, 100.2, 100.1, 100.7, 101.1, 100.9, 100.4],
            "SOXX": [200, 201, 203, 202, 201, 200, 201, 202, 201, 200],
        },
        index=asset_prices.index,
    )

    result = run_residual_backtest(
        asset_prices=asset_prices,
        weights={"AAA": 1.0, "BBB": -1.0},
        factor_prices=factor_prices,
        config=ResidualStrategyConfig(
            factor_symbols=("QQQ", "SOXX"),
            entry_z=0.5,
            exit_z=0.0,
            signal_lookback=3,
            beta_lookback=3,
            transaction_cost_bps=10.0,
            borrow_cost_bps=50.0,
        ),
        basket_label="residual basket",
    )

    assert result.strategy == "factor_neutral_residual"
    assert {"beta_QQQ", "beta_SOXX", "transaction_cost", "borrow_cost"}.issubset(
        result.timeseries.columns
    )
    assert {
        "vol_target_scale",
        "raw_position",
        "filtered_position",
        "regime_ok",
        "residual_vol_ok",
        "factor_vol_ok",
        "beta_stability_ok",
    }.issubset(result.timeseries.columns)
    assert result.timeseries["transaction_cost"].sum() >= 0


def test_residual_momentum_backtest_runs() -> None:
    asset_prices = pd.DataFrame(
        {
            "AAA": [100, 102, 103, 105, 106, 107, 108, 110, 109, 111],
            "BBB": [100, 99, 98, 97, 96, 95, 96, 97, 98, 99],
        },
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    factor_prices = pd.DataFrame(
        {
            "SPY": [100, 100.1, 100.2, 100.3, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8],
            "QQQ": [100, 100.3, 100.5, 100.8, 101.0, 101.1, 101.3, 101.6, 101.7, 101.9],
        },
        index=asset_prices.index,
    )

    result = run_residual_momentum_backtest(
        asset_prices=asset_prices,
        weights={"AAA": 1.0, "BBB": -1.0},
        factor_prices=factor_prices,
        config=ResidualStrategyConfig(
            factor_symbols=("SPY", "QQQ"),
            entry_z=0.5,
            exit_z=0.0,
            signal_lookback=3,
            beta_lookback=3,
        ),
        basket_label="momentum basket",
    )

    assert result.strategy == "factor_neutral_residual_momentum"
    assert "portfolio_returns" in result.timeseries.columns
