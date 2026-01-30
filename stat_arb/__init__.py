"""Shared utilities for basket discovery and backtesting."""

from .attribution import (
    build_backtest_attribution_summary,
    build_walk_forward_selection_attribution,
    write_backtest_attribution_files,
    write_walk_forward_attribution_file,
)
from .backtests import (
    BacktestResult,
    MeanReversionConfig,
    PortfolioConfig,
    ResidualStrategyConfig,
    run_mean_reversion_backtest,
    run_residual_backtest,
    run_residual_momentum_backtest,
    run_weighted_portfolio_backtest,
)
from .baskets import (
    BasketParseError,
    BasketSpec,
    parse_basket_string,
    parse_weights_json,
    slugify_trade,
)
from .events import (
    build_catalyst_signal,
    build_event_blackout_mask,
    load_event_calendar,
)
from .factors import FACTOR_PRESETS, resolve_factor_symbols
from .prices import DEFAULT_CACHE_DIRS, load_price_panel, resolve_cache_dirs
from .walkforward import (
    CandidateBasket,
    WalkForwardConfig,
    WalkForwardResult,
    load_candidate_baskets,
    run_walk_forward_selection,
)

__all__ = [
    "BacktestResult",
    "BasketParseError",
    "BasketSpec",
    "DEFAULT_CACHE_DIRS",
    "build_event_blackout_mask",
    "build_catalyst_signal",
    "build_backtest_attribution_summary",
    "build_walk_forward_selection_attribution",
    "FACTOR_PRESETS",
    "load_event_calendar",
    "MeanReversionConfig",
    "PortfolioConfig",
    "ResidualStrategyConfig",
    "CandidateBasket",
    "load_price_panel",
    "load_candidate_baskets",
    "parse_basket_string",
    "parse_weights_json",
    "resolve_cache_dirs",
    "run_mean_reversion_backtest",
    "run_residual_backtest",
    "run_residual_momentum_backtest",
    "run_walk_forward_selection",
    "run_weighted_portfolio_backtest",
    "slugify_trade",
    "WalkForwardConfig",
    "WalkForwardResult",
    "resolve_factor_symbols",
    "write_backtest_attribution_files",
    "write_walk_forward_attribution_file",
]
