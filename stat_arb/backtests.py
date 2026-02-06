from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .metrics import (
    annualized_return,
    annualized_vol,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


@dataclass(frozen=True)
class PortfolioConfig:
    rebalance: Literal["daily", "weekly", "monthly", "none"] = "daily"


@dataclass(frozen=True)
class MeanReversionConfig:
    entry_z: float = 2.0
    exit_z: float = 0.0
    lookback: int = 60


@dataclass(frozen=True)
class ResidualStrategyConfig:
    factor_symbols: tuple[str, ...] = ("SPY", "QQQ", "XLK", "SMH", "SOXX")
    entry_z: float = 2.0
    exit_z: float = 0.0
    signal_lookback: int = 60
    beta_lookback: int = 126
    hedge_rebalance: Literal["daily", "weekly", "monthly"] = "weekly"
    target_vol: float = 0.12
    vol_lookback: int = 20
    max_gross_leverage: float = 1.5
    max_residual_vol: float = 0.35
    beta_stability_lookback: int = 20
    max_beta_instability: float = 0.15
    max_factor_vol: float = 0.35
    transaction_cost_bps: float = 5.0
    borrow_cost_bps: float = 30.0


@dataclass
class BacktestResult:
    strategy: str
    weights: dict[str, float]
    prices: pd.DataFrame
    timeseries: pd.DataFrame
    stats: pd.Series
    basket_label: str | None = None


def _normalize_weights(weights: dict[str, float]) -> pd.Series:
    weight_vec = pd.Series(weights, dtype=float)
    gross = weight_vec.abs().sum()
    if gross <= 0:
        raise ValueError("Gross exposure must be positive.")
    return weight_vec / gross


def _signal_to_position(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    mode: Literal["mean_reversion", "momentum"] = "mean_reversion",
) -> pd.Series:
    signal = pd.Series(0.0, index=zscore.index, name="signal")
    state = 0.0

    for idx, z_value in enumerate(zscore):
        if np.isnan(z_value):
            signal.iloc[idx] = state
            continue
        if state == 0.0:
            if mode == "mean_reversion":
                if z_value > entry_z:
                    state = -1.0
                elif z_value < -entry_z:
                    state = 1.0
            else:
                if z_value > entry_z:
                    state = 1.0
                elif z_value < -entry_z:
                    state = -1.0
        elif mode == "mean_reversion":
            if state == 1.0 and z_value >= exit_z:
                state = 0.0
            elif state == -1.0 and z_value <= exit_z:
                state = 0.0
        else:
            if state == 1.0 and z_value <= exit_z:
                state = 0.0
            elif state == -1.0 and z_value >= -exit_z:
                state = 0.0
        signal.iloc[idx] = state

    position = signal.shift(1).fillna(0.0)
    position.name = "position"
    return position


def _apply_blackout(position: pd.Series, blackout_mask: pd.Series | None) -> pd.Series:
    if blackout_mask is None:
        return position
    adjusted = position.copy()
    aligned_mask = blackout_mask.reindex(position.index).fillna(False).astype(bool)
    adjusted.loc[aligned_mask] = 0.0
    adjusted.name = position.name
    return adjusted


def _apply_catalyst_alignment(
    position: pd.Series,
    catalyst_signal: pd.Series | None,
    mode: Literal["mean_reversion", "momentum"],
    require_catalyst: bool,
) -> pd.Series:
    if catalyst_signal is None:
        return position

    aligned = position.copy()
    catalyst_signal = catalyst_signal.reindex(position.index).fillna(0.0)
    if mode == "momentum":
        if require_catalyst:
            aligned.loc[(aligned > 0) & (catalyst_signal <= 0)] = 0.0
            aligned.loc[(aligned < 0) & (catalyst_signal >= 0)] = 0.0
        else:
            aligned.loc[(aligned > 0) & (catalyst_signal < 0)] = 0.0
            aligned.loc[(aligned < 0) & (catalyst_signal > 0)] = 0.0
    else:
        aligned.loc[catalyst_signal != 0] = 0.0
    aligned.name = position.name
    return aligned


def _vol_target_scale(
    returns: pd.Series,
    target_vol: float,
    lookback: int,
    max_gross_leverage: float,
) -> pd.Series:
    realized_vol = returns.rolling(lookback).std(ddof=0) * np.sqrt(252)
    scale = (target_vol / realized_vol.replace(0.0, np.nan)).shift(1)
    scale = scale.clip(lower=0.0, upper=max_gross_leverage).fillna(0.0)
    scale.name = "vol_target_scale"
    return scale


def _regime_filter_mask(
    residual_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factor_betas: pd.DataFrame,
    config: ResidualStrategyConfig,
) -> pd.DataFrame:
    residual_vol = residual_returns.rolling(config.vol_lookback).std(ddof=0) * np.sqrt(
        252
    )
    factor_portfolio_vol = factor_returns.rolling(config.vol_lookback).std(
        ddof=0
    ) * np.sqrt(252)
    beta_instability = factor_betas.rolling(config.beta_stability_lookback).std(ddof=0)

    filters = pd.DataFrame(index=residual_returns.index)
    filters["residual_vol_ok"] = (residual_vol <= config.max_residual_vol).fillna(False)
    filters["factor_vol_ok"] = (
        factor_portfolio_vol.max(axis=1) <= config.max_factor_vol
    ).fillna(False)
    filters["beta_stability_ok"] = (
        beta_instability.max(axis=1) <= config.max_beta_instability
    ).fillna(False)
    filters["regime_ok"] = filters.all(axis=1)
    filters["residual_vol"] = residual_vol
    filters["max_factor_vol"] = factor_portfolio_vol.max(axis=1)
    filters["max_beta_instability"] = beta_instability.max(axis=1)
    return filters


def _recompute_equity(returns: pd.Series) -> pd.Series:
    equity_curve = (1.0 + returns.fillna(0.0)).cumprod()
    equity_curve.name = "equity_curve"
    return equity_curve


def _asset_contribution_frame(
    prices: pd.DataFrame,
    weights: dict[str, float],
    exposure: pd.Series,
) -> pd.DataFrame:
    weight_vec = _normalize_weights(weights)
    asset_returns = prices[list(weight_vec.index)].pct_change().fillna(0.0)
    contribution_frame = asset_returns.mul(weight_vec, axis=1).mul(exposure, axis=0)
    contribution_frame = contribution_frame.rename(
        columns={
            symbol: f"asset_contrib_{symbol}" for symbol in contribution_frame.columns
        }
    )
    return contribution_frame


def compute_basket_notional(
    prices: pd.DataFrame, weights: dict[str, float]
) -> pd.Series:
    missing = [symbol for symbol in weights if symbol not in prices.columns]
    if missing:
        raise ValueError(f"Missing symbols in prices: {missing}")

    basket = pd.Series(0.0, index=prices.index)
    for symbol, weight in weights.items():
        basket = basket.add(weight * prices[symbol], fill_value=0.0)
    basket.name = "basket_notional"
    return basket


def compute_portfolio_from_returns(
    prices: pd.DataFrame,
    weights: dict[str, float],
    rebalance: Literal["daily", "weekly", "monthly", "none"],
) -> tuple[pd.Series, pd.Series]:
    returns = prices.pct_change().fillna(0.0)
    weight_vec = _normalize_weights(weights)

    if rebalance == "daily":
        portfolio_returns = (
            returns[list(weight_vec.index)].mul(weight_vec, axis=1).sum(axis=1)
        )
    elif rebalance == "none":
        shares = weight_vec / prices.iloc[0][weight_vec.index]
        basket_value = prices[weight_vec.index].mul(shares, axis=1).sum(axis=1)
        portfolio_returns = basket_value.pct_change().fillna(0.0)
    else:
        freq_map = {"weekly": "W-FRI", "monthly": "M"}
        if rebalance not in freq_map:
            raise ValueError(f"Unsupported rebalance frequency: {rebalance}")

        anchors = prices.resample(freq_map[rebalance]).last().index
        anchors = pd.Index(sorted(set(anchors).intersection(prices.index)))
        if len(anchors) == 0 or anchors[0] != prices.index[0]:
            anchors = anchors.insert(0, prices.index[0])

        portfolio_returns = pd.Series(index=prices.index, dtype=float)
        current_shares = None
        for idx, start in enumerate(anchors):
            end = anchors[idx + 1] if idx + 1 < len(anchors) else None
            start_loc = prices.index.get_loc(start)
            if isinstance(start_loc, slice):
                start_loc = start_loc.start
            subset = prices.iloc[start_loc:] if end is None else prices.loc[start:end]
            if current_shares is None or start in anchors:
                base_prices = prices.loc[start, weight_vec.index]
                current_shares = weight_vec / base_prices
            basket_value = (
                subset[weight_vec.index].mul(current_shares, axis=1).sum(axis=1)
            )
            subset_returns = basket_value.pct_change().fillna(0.0)
            if len(portfolio_returns.loc[subset.index].dropna()) > 0:
                subset_returns = subset_returns.iloc[1:]
            portfolio_returns.loc[subset_returns.index] = subset_returns
        portfolio_returns = portfolio_returns.fillna(0.0)

    equity_curve = (1.0 + portfolio_returns).cumprod()
    portfolio_returns.name = "portfolio_returns"
    equity_curve.name = "equity_curve"
    return portfolio_returns, equity_curve


def compute_mean_reversion_strategy(
    prices: pd.DataFrame,
    weights: dict[str, float],
    config: MeanReversionConfig,
    blackout_mask: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    prices = prices.copy().dropna()
    weight_vec = _normalize_weights(weights)

    log_prices = np.log(prices.replace(0, np.nan)).ffill().dropna()
    spread = log_prices[list(weight_vec.index)].mul(weight_vec, axis=1).sum(axis=1)
    spread.name = "basket_notional"

    rolling_mean = spread.rolling(config.lookback).mean()
    rolling_std = spread.rolling(config.lookback).std(ddof=0)
    zscore = ((spread - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)
    zscore.name = "spread_z"

    asset_returns = prices[list(weight_vec.index)].pct_change().fillna(0.0)
    raw_spread_returns = asset_returns.mul(weight_vec, axis=1).sum(axis=1)

    raw_position = _signal_to_position(
        zscore,
        entry_z=config.entry_z,
        exit_z=config.exit_z,
    )
    position = _apply_blackout(raw_position, blackout_mask)

    strategy_returns = (position * raw_spread_returns).fillna(0.0)
    strategy_returns.name = "portfolio_returns"

    equity_curve = _recompute_equity(strategy_returns)

    return spread, zscore, position, strategy_returns, equity_curve


def _rolling_factor_hedge(
    target_returns: pd.Series,
    factor_returns: pd.DataFrame,
    lookback: int,
    rebalance: Literal["daily", "weekly", "monthly"],
) -> tuple[pd.Series, pd.DataFrame]:
    intercept = pd.Series(
        index=target_returns.index, dtype=float, name="hedge_intercept"
    )
    betas = pd.DataFrame(
        index=target_returns.index, columns=factor_returns.columns, dtype=float
    )

    y_values = target_returns.to_numpy(dtype=float)
    x_values = factor_returns.to_numpy(dtype=float)

    if rebalance == "daily":
        refit_mask = pd.Series(True, index=target_returns.index)
    else:
        freq_map = {"weekly": "W-FRI", "monthly": "M"}
        anchors = factor_returns.resample(freq_map[rebalance]).last().index
        refit_mask = pd.Series(False, index=target_returns.index)
        refit_mask.loc[target_returns.index.intersection(anchors)] = True
        if not refit_mask.iloc[0]:
            refit_mask.iloc[0] = True

    for end_idx in range(lookback, len(target_returns)):
        if not bool(refit_mask.iloc[end_idx]):
            continue
        y_window = y_values[end_idx - lookback : end_idx]
        x_window = x_values[end_idx - lookback : end_idx]
        if not np.isfinite(y_window).all() or not np.isfinite(x_window).all():
            continue
        design = np.column_stack([np.ones(len(x_window)), x_window])
        coef, *_ = np.linalg.lstsq(design, y_window, rcond=None)
        intercept.iloc[end_idx] = coef[0]
        betas.iloc[end_idx] = coef[1:]

    intercept = intercept.ffill().fillna(0.0)
    betas = betas.ffill().fillna(0.0)
    return intercept, betas


