import pandas as pd

from stat_arb.optimizer import PortfolioOptimizerConfig, optimize_candidate_portfolio


def test_optimize_candidate_portfolio_respects_caps() -> None:
    validation_returns = pd.DataFrame(
        {
            "a": [0.01, 0.0, 0.02, -0.01, 0.01],
            "b": [0.009, 0.0, 0.019, -0.011, 0.009],
            "c": [0.002, 0.001, 0.0, 0.001, 0.002],
        }
    )
    selection_scores = pd.Series({"a": 1.0, "b": 0.9, "c": 0.3})
    archetypes = pd.Series({"a": "same", "b": "same", "c": "other"})

    weights = optimize_candidate_portfolio(
        validation_returns=validation_returns,
        selection_scores=selection_scores,
        archetypes=archetypes,
        config=PortfolioOptimizerConfig(max_weight=0.55, max_archetype_weight=0.7),
    )

    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert float(weights.max()) <= 0.55 + 1e-9
    assert float(weights.loc[["a", "b"]].sum()) <= 0.7 + 1e-9
