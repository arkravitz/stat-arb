from pathlib import Path

import pandas as pd

from stat_arb.backtests import MeanReversionConfig, run_mean_reversion_backtest
from stat_arb.walkforward import (
    CandidateBasket,
    WalkForwardConfig,
    load_candidate_baskets,
    run_walk_forward_selection,
)


def test_load_candidate_baskets_prefers_weights_and_dedupes(tmp_path: Path) -> None:
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text(
        "trade,weights\n"
        "\"long 1 units of AAA, short 1 units of BBB\",\"{'AAA': 1, 'BBB': -1}\"\n"
        "\"long 2 units of AAA, short 2 units of BBB\",\"{'AAA': 1, 'BBB': -1}\"\n",
        encoding="utf-8",
    )

    candidates = load_candidate_baskets(csv_path)

    assert len(candidates) == 1
    assert candidates[0].weights == {"AAA": 1.0, "BBB": -1.0}


def test_walk_forward_selection_outputs_oos_timeseries() -> None:
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [100 + (i % 7) for i in range(120)],
            "BBB": [100 + ((i * 2) % 9) for i in range(120)],
        },
        index=dates,
    )
    candidate = CandidateBasket(
        trade="long 1 units of AAA, short 1 units of BBB",
        weights={"AAA": 1.0, "BBB": -1.0},
        source_row={"trade": "long 1 units of AAA, short 1 units of BBB"},
    )

    def runner(history: pd.DataFrame, current: CandidateBasket):
        return run_mean_reversion_backtest(
            prices=history[list(current.weights.keys())],
            weights=current.weights,
            config=MeanReversionConfig(entry_z=0.5, exit_z=0.0, lookback=5),
            basket_label=current.trade,
        )

    result = run_walk_forward_selection(
        prices=prices,
        candidates=[candidate],
        config=WalkForwardConfig(
            train_days=40,
            validate_days=20,
            test_days=20,
            top_k=1,
            min_train_sharpe=-10.0,
            min_validation_sharpe=-10.0,
            min_validation_entries=0,
        ),
        runner=runner,
        warmup_bars=5,
    )

    assert not result.timeseries.empty
    assert {
        "portfolio_returns",
        "equity_curve",
        "selected_count",
        "capital_used",
    }.issubset(result.timeseries.columns)
    assert not result.fold_summary.empty
    assert "portfolio_weight" in result.fold_summary.columns


def test_walk_forward_selection_enforces_symbol_diversity() -> None:
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [100 + (i % 5) for i in range(180)],
            "BBB": [98 + ((i * 2) % 7) for i in range(180)],
            "CCC": [101 + ((i * 3) % 6) for i in range(180)],
            "DDD": [99 + ((i * 4) % 8) for i in range(180)],
        },
        index=dates,
    )
    candidates = [
        CandidateBasket(
            trade="c1",
            weights={"AAA": 1.0, "BBB": -1.0},
            source_row={"trade": "c1"},
        ),
        CandidateBasket(
            trade="c2",
            weights={"AAA": 1.0, "BBB": -1.0, "CCC": 1.0},
            source_row={"trade": "c2"},
        ),
        CandidateBasket(
            trade="c3",
            weights={"CCC": 1.0, "DDD": -1.0},
            source_row={"trade": "c3"},
        ),
    ]

    def runner(history: pd.DataFrame, current: CandidateBasket):
        return run_mean_reversion_backtest(
            prices=history[list(current.weights.keys())],
            weights=current.weights,
            config=MeanReversionConfig(entry_z=0.5, exit_z=0.0, lookback=5),
            basket_label=current.trade,
        )

    result = run_walk_forward_selection(
        prices=prices,
        candidates=candidates,
        config=WalkForwardConfig(
            train_days=60,
            validate_days=30,
            test_days=30,
            top_k=2,
            min_train_sharpe=-10.0,
            min_validation_sharpe=-10.0,
            min_validation_entries=0,
            max_symbol_overlap=0.5,
            max_validation_correlation=0.99,
        ),
        runner=runner,
        warmup_bars=5,
    )

    fold_one = result.fold_summary[result.fold_summary["fold"] == 1]
    assert len(fold_one) == 2
    assert not {"c1", "c2"}.issubset(set(fold_one["trade"]))
