import math

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def annualized_return(daily_returns: pd.Series) -> float:
    periods = len(daily_returns)
    if periods == 0:
        return np.nan
    total = (1.0 + daily_returns).prod()
    return total ** (TRADING_DAYS / periods) - 1.0


def annualized_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std(ddof=0) * math.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    vol = annualized_vol(daily_returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    excess = annualized_return(daily_returns) - rf
    return excess / vol


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    downside = daily_returns[daily_returns < 0].std(ddof=0) * math.sqrt(TRADING_DAYS)
    if downside == 0 or np.isnan(downside):
        return np.nan
    excess = annualized_return(daily_returns) - rf
    return excess / downside


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    return drawdown.min()


def calmar_ratio(daily_returns: pd.Series, equity_curve: pd.Series) -> float:
    drawdown = abs(max_drawdown(equity_curve))
    if drawdown == 0 or np.isnan(drawdown):
        return np.nan
    return annualized_return(daily_returns) / drawdown
