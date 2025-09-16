"""
plot_trades_utils.py – optional visual helpers
Place inside: stock-bot/src/
"""

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

def plot_equity_curve(equity_dict: dict):
    """
    Show equity curves in a single window.
    equity_dict example: {"SAC": pd.Series, "DQN": pd.Series}
    """
    plt.figure(figsize=(10, 6))
    for label, eq in equity_dict.items():
        plt.plot(eq.index, eq.values, label=label)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Equity Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_candlestick_with_trades(price_df: pd.DataFrame,
                                 trades: pd.DataFrame):
    """
    price_df: columns = ['Open','High','Low','Close','Volume']
    trades  : DataFrame with columns ['Date','Action'] (BUY/SELL)
    Displays chart inline instead of saving to file.
    """
    buy_dates  = trades[trades.Action == "BUY"]["Date"]
    sell_dates = trades[trades.Action == "SELL"]["Date"]

    buy_series  = pd.Series(price_df['Close'], index=price_df.index)\
                     .where(price_df.index.isin(buy_dates))
    sell_series = pd.Series(price_df['Close'], index=price_df.index)\
                     .where(price_df.index.isin(sell_dates))

    ap = [
        mpf.make_addplot(buy_series, type='scatter', markersize=80,
                         marker='^', color='g'),
        mpf.make_addplot(sell_series, type='scatter', markersize=80,
                         marker='v', color='r')
    ]
    mpf.plot(price_df, type='candle', volume=True,
             addplot=ap, style='yahoo', title='Trades Overlay')
