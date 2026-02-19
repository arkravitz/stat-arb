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


def compute_residual_mean_reversion_strategy(
    asset_prices: pd.DataFrame,
    weights: dict[str, float],
    factor_prices: pd.DataFrame,
    config: ResidualStrategyConfig,
    blackout_mask: pd.Series | None = None,
    catalyst_signal: pd.Series | None = None,
    require_catalyst: bool = False,
    mode: Literal["mean_reversion", "momentum"] = "mean_reversion",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    combined_prices = pd.concat([asset_prices, factor_prices], axis=1).dropna()
    asset_prices = combined_prices[list(asset_prices.columns)]
    factor_prices = combined_prices[list(factor_prices.columns)]
    weight_vec = _normalize_weights(weights)

    asset_returns = asset_prices[list(weight_vec.index)].pct_change().fillna(0.0)
    factor_returns = factor_prices.pct_change().fillna(0.0)
    raw_spread_returns = asset_returns.mul(weight_vec, axis=1).sum(axis=1)
    raw_spread_returns.name = "raw_spread_returns"

    hedge_intercept, factor_betas = _rolling_factor_hedge(
        target_returns=raw_spread_returns,
        factor_returns=factor_returns,
        lookback=config.beta_lookback,
        rebalance=config.hedge_rebalance,
    )
    hedged_factor_returns = factor_returns.mul(factor_betas, axis=1).sum(axis=1)
    residual_returns = (
        raw_spread_returns - hedge_intercept - hedged_factor_returns
    ).fillna(0.0)
    residual_returns.name = "residual_returns"

    residual_level = residual_returns.cumsum()
    residual_level.name = "basket_notional"

    rolling_mean = residual_level.rolling(config.signal_lookback).mean()
    rolling_std = residual_level.rolling(config.signal_lookback).std(ddof=0)
    zscore = ((residual_level - rolling_mean) / rolling_std).replace(
        [np.inf, -np.inf], np.nan
    )
    zscore.name = "spread_z"

    raw_position = _signal_to_position(
        zscore,
        entry_z=config.entry_z,
        exit_z=config.exit_z,
        mode=mode,
    )
    raw_position = _apply_blackout(raw_position, blackout_mask)
    raw_position = _apply_catalyst_alignment(
        raw_position,
        catalyst_signal=catalyst_signal,
        mode=mode,
        require_catalyst=require_catalyst,
    )
    regime_filters = _regime_filter_mask(
        residual_returns=residual_returns,
        factor_returns=factor_returns,
        factor_betas=factor_betas,
        config=config,
    )
    filtered_position = raw_position.where(regime_filters["regime_ok"], 0.0)
    vol_scale = _vol_target_scale(
        residual_returns,
        target_vol=config.target_vol,
        lookback=config.vol_lookback,
        max_gross_leverage=config.max_gross_leverage,
    )
    position = (filtered_position * vol_scale).rename("position")

    turnover = position.diff().abs().fillna(position.abs())
    total_gross_exposure = 1.0 + factor_betas.abs().sum(axis=1)
    basket_long = float(weight_vec.clip(lower=0).sum())
    basket_short = float((-weight_vec.clip(upper=0)).sum())
    factor_short_long_spread = factor_betas.clip(lower=0).sum(axis=1)
    factor_short_short_spread = (-factor_betas.clip(upper=0)).sum(axis=1)
    short_exposure = pd.Series(0.0, index=position.index, name="short_exposure")
    short_exposure[position > 0] = basket_short + factor_short_long_spread[position > 0]
    short_exposure[position < 0] = basket_long + factor_short_short_spread[position < 0]

    transaction_cost = (
        turnover * total_gross_exposure * (config.transaction_cost_bps / 10000.0)
    )
    borrow_cost = (
        short_exposure * abs(position) * (config.borrow_cost_bps / 10000.0 / 252.0)
    )
    gross_returns = position * residual_returns
    strategy_returns = (gross_returns - transaction_cost - borrow_cost).fillna(0.0)
    strategy_returns.name = "portfolio_returns"

    diagnostics = factor_betas.add_prefix("beta_")
    diagnostics = pd.concat(
        [
            diagnostics,
            factor_returns.rename(columns=lambda symbol: f"factor_ret_{symbol}"),
            factor_returns.mul(-position, axis=0)
            .mul(factor_betas, axis=1)
            .rename(columns=lambda symbol: f"factor_contrib_{symbol}"),
        ],
        axis=1,
    )
    diagnostics["hedge_intercept"] = hedge_intercept
    diagnostics["gross_spread_returns"] = gross_returns
    diagnostics["transaction_cost"] = transaction_cost
    diagnostics["borrow_cost"] = borrow_cost
    diagnostics["gross_exposure"] = total_gross_exposure
    diagnostics["short_exposure"] = short_exposure
    diagnostics["vol_target_scale"] = vol_scale
    diagnostics["raw_position"] = raw_position
    if catalyst_signal is not None:
        diagnostics["catalyst_signal"] = catalyst_signal.reindex(position.index).fillna(
            0.0
        )
    diagnostics["filtered_position"] = filtered_position
    diagnostics = pd.concat([diagnostics, regime_filters], axis=1)
    if blackout_mask is not None:
        diagnostics["event_blackout"] = (
            blackout_mask.reindex(position.index).fillna(False).astype(int)
        )
    diagnostics["signal_mode"] = 1 if mode == "momentum" else -1

    equity_curve = _recompute_equity(strategy_returns)
    return residual_level, zscore, position, strategy_returns, equity_curve, diagnostics


def compute_stats(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    basket_notional: pd.Series | None = None,
    zscore: pd.Series | None = None,
    position: pd.Series | None = None,
) -> pd.Series:
    stats = {
        "start": str(daily_returns.index.min().date()),
        "end": str(daily_returns.index.max().date()),
        "days": int(len(daily_returns)),
        "total_return": float(equity_curve.iloc[-1] - 1.0),
        "annual_return": float(annualized_return(daily_returns)),
        "annual_vol": float(annualized_vol(daily_returns)),
        "sharpe": float(sharpe_ratio(daily_returns)),
        "sortino": float(sortino_ratio(daily_returns)),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "calmar": float(calmar_ratio(daily_returns, equity_curve)),
        "win_rate": float((daily_returns > 0).mean()),
        "avg_daily_return": float(daily_returns.mean()),
        "best_day": float(daily_returns.max()),
        "worst_day": float(daily_returns.min()),
        "skew": float(daily_returns.skew()),
        "kurtosis": float(daily_returns.kurtosis()),
    }

    if basket_notional is not None and basket_notional.std(ddof=0) > 0:
        centered = (basket_notional - basket_notional.mean()) / basket_notional.std(
            ddof=0
        )
        stats["basket_mean"] = float(basket_notional.mean())
        stats["basket_std"] = float(basket_notional.std(ddof=0))
        stats["basket_last_zscore"] = float(centered.iloc[-1])

    if zscore is not None:
        valid_zscore = zscore.dropna()
        if not valid_zscore.empty:
            stats["spread_z_last"] = float(valid_zscore.iloc[-1])
            stats["spread_z_abs_mean"] = float(valid_zscore.abs().mean())

    if position is not None:
        entries = int(((position != 0) & (position.shift(1).fillna(0) == 0)).sum())
        exits = int(((position == 0) & (position.shift(1).fillna(0) != 0)).sum())
        active_days = int((position != 0).sum())

        trade_lengths: list[int] = []
        current_length = 0
        for current_position in position.fillna(0.0):
            if current_position != 0:
                current_length += 1
            elif current_length > 0:
                trade_lengths.append(current_length)
                current_length = 0
        if current_length > 0:
            trade_lengths.append(current_length)

        stats["entries"] = entries
        stats["exits"] = exits
        stats["active_days"] = active_days
        stats["active_fraction"] = (
            float(active_days / len(position)) if len(position) else np.nan
        )
        stats["avg_holding_days"] = (
            float(np.mean(trade_lengths)) if trade_lengths else np.nan
        )

    return pd.Series(stats)


def run_weighted_portfolio_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    config: PortfolioConfig = PortfolioConfig(),
    basket_label: str | None = None,
) -> BacktestResult:
    prices = prices.dropna()
    if prices.empty:
        raise RuntimeError(
            "No overlapping price history found after filtering dates and dropping NaNs."
        )

    basket_notional = compute_basket_notional(prices, weights)
    daily_returns, equity_curve = compute_portfolio_from_returns(
        prices, weights, config.rebalance
    )
    contributions = _asset_contribution_frame(
        prices=prices,
        weights=weights,
        exposure=pd.Series(1.0, index=prices.index, name="exposure"),
    )
    stats = compute_stats(daily_returns, equity_curve, basket_notional=basket_notional)
    timeseries = pd.concat(
        [prices, basket_notional, daily_returns, equity_curve, contributions], axis=1
    )

    return BacktestResult(
        strategy="weighted_portfolio",
        weights=weights,
        prices=prices,
        timeseries=timeseries,
        stats=stats,
        basket_label=basket_label,
    )


def run_mean_reversion_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    config: MeanReversionConfig = MeanReversionConfig(),
    basket_label: str | None = None,
    blackout_mask: pd.Series | None = None,
) -> BacktestResult:
    prices = prices.dropna()
    if prices.empty:
        raise RuntimeError(
            "No overlapping price history found after filtering dates and dropping NaNs."
        )

    spread, zscore, position, daily_returns, equity_curve = (
        compute_mean_reversion_strategy(
            prices=prices,
            weights=weights,
            config=config,
            blackout_mask=blackout_mask,
        )
    )
    stats = compute_stats(
        daily_returns,
        equity_curve,
        basket_notional=spread,
        zscore=zscore,
        position=position,
    )
    contributions = _asset_contribution_frame(
        prices=prices,
        weights=weights,
        exposure=position,
    )
    timeseries = pd.concat(
        [prices, spread, zscore, position, daily_returns, equity_curve, contributions],
        axis=1,
    )

    return BacktestResult(
        strategy="mean_reversion",
        weights=weights,
        prices=prices,
        timeseries=timeseries,
        stats=stats,
        basket_label=basket_label,
    )


def run_residual_backtest(
    asset_prices: pd.DataFrame,
    weights: dict[str, float],
    factor_prices: pd.DataFrame,
    config: ResidualStrategyConfig = ResidualStrategyConfig(),
    basket_label: str | None = None,
    blackout_mask: pd.Series | None = None,
    catalyst_signal: pd.Series | None = None,
    require_catalyst: bool = False,
) -> BacktestResult:
    asset_prices = asset_prices.dropna()
    factor_prices = factor_prices.dropna()
    if asset_prices.empty or factor_prices.empty:
        raise RuntimeError(
            "No overlapping asset/factor history found after filtering dates and dropping NaNs."
        )

    residual_level, zscore, position, daily_returns, equity_curve, diagnostics = (
        compute_residual_mean_reversion_strategy(
            asset_prices=asset_prices,
            weights=weights,
            factor_prices=factor_prices,
            config=config,
            blackout_mask=blackout_mask,
            catalyst_signal=catalyst_signal,
            require_catalyst=require_catalyst,
            mode="mean_reversion",
        )
    )
    aligned_prices = pd.concat([asset_prices, factor_prices], axis=1).dropna()
    aligned_asset_prices = aligned_prices[list(asset_prices.columns)]
    stats = compute_stats(
        daily_returns,
        equity_curve,
        basket_notional=residual_level,
        zscore=zscore,
        position=position,
    )
    contributions = _asset_contribution_frame(
        prices=aligned_asset_prices,
        weights=weights,
        exposure=position,
    )
    timeseries = pd.concat(
        [
            aligned_asset_prices,
            residual_level,
            zscore,
            position,
            daily_returns,
            equity_curve,
            contributions,
            diagnostics,
        ],
        axis=1,
    )

    return BacktestResult(
        strategy="factor_neutral_residual",
        weights=weights,
        prices=aligned_asset_prices,
        timeseries=timeseries,
        stats=stats,
        basket_label=basket_label,
    )


def run_residual_momentum_backtest(
    asset_prices: pd.DataFrame,
    weights: dict[str, float],
    factor_prices: pd.DataFrame,
    config: ResidualStrategyConfig = ResidualStrategyConfig(),
    basket_label: str | None = None,
    blackout_mask: pd.Series | None = None,
    catalyst_signal: pd.Series | None = None,
    require_catalyst: bool = False,
) -> BacktestResult:
    asset_prices = asset_prices.dropna()
    factor_prices = factor_prices.dropna()
    if asset_prices.empty or factor_prices.empty:
        raise RuntimeError(
            "No overlapping asset/factor history found after filtering dates and dropping NaNs."
        )

    residual_level, zscore, position, daily_returns, equity_curve, diagnostics = (
        compute_residual_mean_reversion_strategy(
            asset_prices=asset_prices,
            weights=weights,
            factor_prices=factor_prices,
            config=config,
            blackout_mask=blackout_mask,
            catalyst_signal=catalyst_signal,
            require_catalyst=require_catalyst,
            mode="momentum",
        )
    )
    aligned_prices = pd.concat([asset_prices, factor_prices], axis=1).dropna()
    aligned_asset_prices = aligned_prices[list(asset_prices.columns)]
    stats = compute_stats(
        daily_returns,
        equity_curve,
        basket_notional=residual_level,
        zscore=zscore,
        position=position,
    )
    contributions = _asset_contribution_frame(
        prices=aligned_asset_prices,
        weights=weights,
        exposure=position,
    )
    timeseries = pd.concat(
        [
            aligned_asset_prices,
            residual_level,
            zscore,
            position,
            daily_returns,
            equity_curve,
            contributions,
            diagnostics,
        ],
        axis=1,
    )

    return BacktestResult(
        strategy="factor_neutral_residual_momentum",
        weights=weights,
        prices=aligned_asset_prices,
        timeseries=timeseries,
        stats=stats,
        basket_label=basket_label,
    )


def slice_backtest_result(
    result: BacktestResult,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> BacktestResult:
    sliced = result.timeseries.loc[start:end].copy()
    if sliced.empty:
        raise RuntimeError("Requested slice produced no rows.")

    sliced["portfolio_returns"] = sliced["portfolio_returns"].fillna(0.0)
    sliced["equity_curve"] = _recompute_equity(sliced["portfolio_returns"])

    basket_notional = (
        sliced["basket_notional"] if "basket_notional" in sliced.columns else None
    )
    zscore = sliced["spread_z"] if "spread_z" in sliced.columns else None
    position = sliced["position"] if "position" in sliced.columns else None
    stats = compute_stats(
        sliced["portfolio_returns"],
        sliced["equity_curve"],
        basket_notional=basket_notional,
        zscore=zscore,
        position=position,
    )
    prices = sliced[result.prices.columns.intersection(sliced.columns)].copy()
    return BacktestResult(
        strategy=result.strategy,
        weights=result.weights,
        prices=prices,
        timeseries=sliced,
        stats=stats,
        basket_label=result.basket_label,
    )
