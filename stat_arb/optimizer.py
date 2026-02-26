from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioOptimizerConfig:
    ridge: float = 1e-3
    score_floor: float = 0.05
    max_weight: float = 0.6
    max_archetype_weight: float = 0.7


def _clip_and_normalize(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = weights.clip(lower=0.0)
    if float(weights.sum()) <= 0:
        weights[:] = 1.0 / len(weights)
    else:
        weights /= float(weights.sum())

    if max_weight <= 0:
        return weights

    for _ in range(10):
        over = weights > max_weight
        if not bool(over.any()):
            break
        excess = float((weights[over] - max_weight).sum())
        weights.loc[over] = max_weight
        under = ~over
        if float(weights.loc[under].sum()) <= 0:
            break
        weights.loc[under] += (
            excess * weights.loc[under] / float(weights.loc[under].sum())
        )
        weights = weights.clip(lower=0.0)
        weights /= float(weights.sum())
    return weights


def optimize_candidate_portfolio(
    validation_returns: pd.DataFrame,
    selection_scores: pd.Series,
    archetypes: pd.Series,
    config: PortfolioOptimizerConfig,
) -> pd.Series:
    validation_returns = validation_returns.fillna(0.0)
    if validation_returns.shape[1] == 1:
        return pd.Series(1.0, index=validation_returns.columns, name="portfolio_weight")

    mu = (
        selection_scores.reindex(validation_returns.columns).fillna(0.0).clip(lower=0.0)
        + config.score_floor
    )
    cov = validation_returns.cov().fillna(0.0)
    cov += np.eye(len(cov)) * config.ridge
    inv_cov = pd.DataFrame(
        np.linalg.pinv(cov.to_numpy(dtype=float)),
        index=cov.index,
        columns=cov.columns,
    )
    raw = inv_cov.dot(mu)
    weights = _clip_and_normalize(raw, config.max_weight)

    if config.max_archetype_weight > 0:
        for _ in range(5):
            changed = False
            for archetype in archetypes.dropna().unique():
                names = list(archetypes[archetypes == archetype].index)
                archetype_weight = float(weights.reindex(names).sum())
                if (
                    archetype_weight > config.max_archetype_weight
                    and archetype_weight > 0
                ):
                    scale = config.max_archetype_weight / archetype_weight
                    weights.loc[names] *= scale
                    changed = True
            if not changed:
                break
            weights = weights.clip(lower=0.0)
            headroom = pd.Series(config.max_weight, index=weights.index) - weights
            headroom = headroom.clip(lower=0.0)
            remaining = 1.0 - float(weights.sum())
            eligible = archetypes.map(
                lambda archetype: float(
                    weights.loc[archetypes[archetypes == archetype].index].sum()
                )
                < config.max_archetype_weight - 1e-9
            )
            headroom = headroom.where(eligible, 0.0)
            if remaining > 0 and float(headroom.sum()) > 0:
                weights += remaining * headroom / float(headroom.sum())
            weights = _clip_and_normalize(weights, config.max_weight)

    weights.name = "portfolio_weight"
    return weights
