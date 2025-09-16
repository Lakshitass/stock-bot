"""
metrics.py – performance metrics for stock or RL strategies
Place this file inside: stock-bot/src/
"""

import numpy as np
import pandas as pd

def total_return(equity: pd.Series) -> float:
    """Total percentage return over the period."""
    return equity.iloc[-1] / equity.iloc[0] - 1

def annualized_return(equity: pd.Series, trading_days: int = 252) -> float:
    """Compound annual growth rate based on trading_days/year."""
    return (equity.iloc[-1] / equity.iloc[0]) ** (trading_days / len(equity)) - 1

def sharpe_ratio(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.
    daily_returns: percent change series (not log).
    """
    excess = daily_returns - risk_free / 252
    return excess.mean() / excess.std(ddof=0) * np.sqrt(252)

def max_drawdown(equity: pd.Series) -> float:
    """Largest percentage drop from a peak."""
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max
    return drawdown.min()
