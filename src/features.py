import pandas as pd
import numpy as np

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral for early rows

def prepare_data(filepath, split_date="2019-01-01"):
    """
    Load, clean, and create features for stock data.
    Returns train and test DataFrames.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Add features
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["RSI14"] = compute_rsi(df["Close"])

    df = df.dropna().reset_index(drop=True)

    train = df[df["Date"] < split_date].reset_index(drop=True)
    test = df[df["Date"] >= split_date].reset_index(drop=True)

    return train, test
