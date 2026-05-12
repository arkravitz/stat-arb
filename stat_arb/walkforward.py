from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .backtests import (
    BacktestResult,
    compute_stats,
    slice_backtest_result,
)
from .baskets import parse_basket_string, parse_weights_json
from .optimizer import PortfolioOptimizerConfig, optimize_candidate_portfolio
from .structure import trade_archetype


@dataclass(frozen=True)
class CandidateBasket:
    trade: str
    weights: dict[str, float]
    source_row: dict[str, str]


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 252
    validate_days: int = 126
    test_days: int = 63
    top_k: int = 2
    min_train_sharpe: float = 0.25
    min_validation_sharpe: float = 0.5
    min_validation_entries: int = 3
    stability_penalty: float = 0.5
    max_symbol_overlap: float = 0.5
    max_validation_correlation: float = 0.8
    correlation_penalty: float = 0.75
    min_validation_vol: float = 1e-4
    max_trade_weight: float = 0.6
    max_archetype_weight: float = 0.7


@dataclass
class WalkForwardResult:
    timeseries: pd.DataFrame
    stats: pd.Series
    fold_summary: pd.DataFrame


def _symbol_overlap_ratio(left: CandidateBasket, right: CandidateBasket) -> float:
    left_symbols = set(left.weights)
    right_symbols = set(right.weights)
    if not left_symbols or not right_symbols:
        return 0.0
    return len(left_symbols.intersection(right_symbols)) / min(
        len(left_symbols), len(right_symbols)
    )


def _validation_correlation(left: BacktestResult, right: BacktestResult) -> float:
    joined = pd.concat(
        [
            left.timeseries["portfolio_returns"].rename("left"),
            right.timeseries["portfolio_returns"].rename("right"),
        ],
        axis=1,
    ).dropna()
    if len(joined) < 10:
        return 0.0
    correlation = joined["left"].corr(joined["right"])
    return float(correlation) if pd.notna(correlation) else 0.0


def _select_diverse_candidates(
    scored: list[
        tuple[float, CandidateBasket, BacktestResult, BacktestResult, BacktestResult]
    ],
    config: WalkForwardConfig,
) -> list[
    tuple[float, CandidateBasket, BacktestResult, BacktestResult, BacktestResult]
]:
    selected: list[
        tuple[float, CandidateBasket, BacktestResult, BacktestResult, BacktestResult]
    ] = []
    for candidate_tuple in scored:
        _, candidate, _, validation_result, _ = candidate_tuple
        keep = True
        for existing in selected:
            _, existing_candidate, _, existing_validation, _ = existing
            if (
                _symbol_overlap_ratio(candidate, existing_candidate)
                > config.max_symbol_overlap
            ):
                keep = False
                break
            if (
                abs(_validation_correlation(validation_result, existing_validation))
                > config.max_validation_correlation
            ):
                keep = False
                break
        if keep:
            selected.append(candidate_tuple)
        if len(selected) >= config.top_k:
            break
    return selected


def _portfolio_weights(
    selected: list[
        tuple[float, CandidateBasket, BacktestResult, BacktestResult, BacktestResult]
    ],
    config: WalkForwardConfig,
) -> pd.Series:
    raw_scores: list[float] = []
    labels: list[str] = []
    archetypes: list[str] = []
    validation_panels: list[pd.Series] = []

    for selection_score, candidate, _, validation_result, _ in selected:
        validation_returns = validation_result.timeseries["portfolio_returns"].dropna()
        validation_vol = float(validation_returns.std(ddof=0))
        if validation_vol <= config.min_validation_vol:
            validation_vol = config.min_validation_vol

        correlation_penalty = 1.0
        correlations: list[float] = []
        for _, other_candidate, _, other_validation, _ in selected:
            if other_candidate.trade == candidate.trade:
                continue
            correlations.append(
                abs(_validation_correlation(validation_result, other_validation))
            )
        if correlations:
            correlation_penalty += config.correlation_penalty * float(
                sum(correlations) / len(correlations)
            )

        score = max(selection_score, 0.0) + 0.1
        raw_scores.append(score / (validation_vol * correlation_penalty))
        labels.append(candidate.trade)
        archetypes.append(trade_archetype(candidate.weights.keys()))
        validation_panels.append(
            validation_result.timeseries["portfolio_returns"].rename(candidate.trade)
        )

    validation_frame = pd.concat(validation_panels, axis=1).fillna(0.0)
    score_series = pd.Series(raw_scores, index=labels, dtype=float)
    archetype_series = pd.Series(archetypes, index=labels, dtype=object)
    return optimize_candidate_portfolio(
        validation_returns=validation_frame,
        selection_scores=score_series,
        archetypes=archetype_series,
        config=PortfolioOptimizerConfig(
            max_weight=config.max_trade_weight,
            max_archetype_weight=config.max_archetype_weight,
        ),
    )


def load_candidate_baskets(csv_path: Path) -> list[CandidateBasket]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    candidates: list[CandidateBasket] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for row in rows:
        trade = (row.get("trade") or "").strip()
        weights_text = (row.get("weights") or "").strip()
        if weights_text:
            spec = parse_weights_json(weights_text)
        elif trade:
            spec = parse_basket_string(trade)
        else:
            continue

        key = tuple(
            sorted(
                (symbol, round(weight, 8)) for symbol, weight in spec.weights.items()
            )
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            CandidateBasket(
                trade=trade
                or ", ".join(
                    f"{symbol}:{weight:.4f}" for symbol, weight in spec.weights.items()
                ),
                weights=spec.weights,
                source_row=row,
            )
        )
    return candidates


