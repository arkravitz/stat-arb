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


def run_walk_forward_selection(
    prices: pd.DataFrame,
    candidates: list[CandidateBasket],
    config: WalkForwardConfig,
    runner: Callable[[pd.DataFrame, CandidateBasket], BacktestResult],
    warmup_bars: int,
) -> WalkForwardResult:
    dates = prices.index
    required = config.train_days + config.validate_days + config.test_days
    if len(dates) < required:
        raise RuntimeError(
            f"Need at least {required} rows for walk-forward evaluation, found {len(dates)}."
        )

    oos_returns: list[pd.Series] = []
    oos_selected_counts: list[pd.Series] = []
    oos_capital_used: list[pd.Series] = []
    fold_rows: list[dict[str, object]] = []

    max_start = len(dates) - required
    for fold_number, fold_start in enumerate(
        range(0, max_start + 1, config.test_days), start=1
    ):
        train_start_idx = fold_start
        train_end_idx = fold_start + config.train_days - 1
        validate_start_idx = train_end_idx + 1
        validate_end_idx = validate_start_idx + config.validate_days - 1
        test_start_idx = validate_end_idx + 1
        test_end_idx = test_start_idx + config.test_days - 1

        warmup_start_idx = max(0, train_start_idx - warmup_bars)
        history = prices.iloc[warmup_start_idx : test_end_idx + 1]

        train_start = dates[train_start_idx]
        train_end = dates[train_end_idx]
        validate_start = dates[validate_start_idx]
        validate_end = dates[validate_end_idx]
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]

        scored: list[
            tuple[
                float, CandidateBasket, BacktestResult, BacktestResult, BacktestResult
            ]
        ] = []
        for candidate in candidates:
            try:
                result = runner(history, candidate)
                train_result = slice_backtest_result(result, train_start, train_end)
                validation_result = slice_backtest_result(
                    result, validate_start, validate_end
                )
                test_result = slice_backtest_result(result, test_start, test_end)
            except Exception:
                continue
            if train_result.stats["sharpe"] < config.min_train_sharpe:
                continue
            if validation_result.stats["sharpe"] < config.min_validation_sharpe:
                continue
            if (
                validation_result.stats.get("entries", 0)
                < config.min_validation_entries
            ):
                continue
            selection_score = float(validation_result.stats["sharpe"]) - (
                config.stability_penalty
                * abs(
                    float(train_result.stats["sharpe"])
                    - float(validation_result.stats["sharpe"])
                )
            )
            scored.append(
                (
                    selection_score,
                    candidate,
                    train_result,
                    validation_result,
                    test_result,
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = _select_diverse_candidates(scored, config)
        if selected:
            weights = _portfolio_weights(selected, config)
            test_returns = []
            for _, candidate, _, _, test_result in selected:
                series = test_result.timeseries["portfolio_returns"].rename(
                    candidate.trade
                )
                test_returns.append(series)
            test_panel = pd.concat(test_returns, axis=1).fillna(0.0)
            fold_returns = test_panel.mul(
                weights.reindex(test_panel.columns), axis=1
            ).sum(axis=1)
            selected_count = pd.Series(
                len(selected), index=fold_returns.index, name="selected_count"
            )
            capital_used = pd.Series(
                float(weights.abs().sum()),
                index=fold_returns.index,
                name="capital_used",
            )
        else:
            fold_returns = pd.Series(
                0.0,
                index=prices.loc[test_start:test_end].index,
                name="portfolio_returns",
            )
            selected_count = pd.Series(
                0.0, index=fold_returns.index, name="selected_count"
            )
            capital_used = pd.Series(0.0, index=fold_returns.index, name="capital_used")

        fold_returns.name = "portfolio_returns"
        oos_returns.append(fold_returns)
        oos_selected_counts.append(selected_count)
        oos_capital_used.append(capital_used)

        if selected:
            for rank, (
                selection_score,
                candidate,
                train_result,
                validation_result,
                test_result,
            ) in enumerate(selected, start=1):
                portfolio_weight = float(weights.get(candidate.trade, 0.0))
                fold_rows.append(
                    {
                        "fold": fold_number,
                        "rank": rank,
                        "trade": candidate.trade,
                        "archetype": trade_archetype(candidate.weights.keys()),
                        "train_start": str(train_start.date()),
                        "train_end": str(train_end.date()),
                        "validate_start": str(validate_start.date()),
                        "validate_end": str(validate_end.date()),
                        "test_start": str(test_start.date()),
                        "test_end": str(test_end.date()),
                        "train_sharpe": float(train_result.stats["sharpe"]),
                        "validation_sharpe": float(validation_result.stats["sharpe"]),
                        "test_sharpe": float(test_result.stats["sharpe"]),
                        "selection_score": float(selection_score),
                        "portfolio_weight": portfolio_weight,
                        "validation_entries": int(
                            validation_result.stats.get("entries", 0)
                        ),
                        "test_entries": int(test_result.stats.get("entries", 0)),
                    }
                )
        else:
            fold_rows.append(
                {
                    "fold": fold_number,
                    "rank": 0,
                    "trade": "",
                    "archetype": "",
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "validate_start": str(validate_start.date()),
                    "validate_end": str(validate_end.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "train_sharpe": float("nan"),
                    "validation_sharpe": float("nan"),
                    "test_sharpe": float("nan"),
                    "selection_score": float("nan"),
                    "portfolio_weight": 0.0,
                    "validation_entries": 0,
                    "test_entries": 0,
                }
            )

    combined_returns = pd.concat(oos_returns).sort_index()
    combined_returns = combined_returns[
        ~combined_returns.index.duplicated(keep="first")
    ]
    combined_returns.name = "portfolio_returns"
    combined_equity = (1.0 + combined_returns).cumprod()
    combined_equity.name = "equity_curve"

    selected_count = pd.concat(oos_selected_counts).sort_index()
    selected_count = selected_count[~selected_count.index.duplicated(keep="first")]
    capital_used = pd.concat(oos_capital_used).sort_index()
    capital_used = capital_used[~capital_used.index.duplicated(keep="first")]

    timeseries = pd.concat(
        [combined_returns, combined_equity, selected_count, capital_used], axis=1
    )
    stats = compute_stats(timeseries["portfolio_returns"], timeseries["equity_curve"])
    return WalkForwardResult(
        timeseries=timeseries,
        stats=stats,
        fold_summary=pd.DataFrame(fold_rows),
    )
