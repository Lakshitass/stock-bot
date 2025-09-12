# src/baseline.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def buy_and_hold(filepath, split_date="2019-01-01", initial_cash=10000):
    df = pd.read_csv(filepath, parse_dates=["Date"]).sort_values("Date")

    # Train/Test split
    train = df[df["Date"] < split_date].reset_index(drop=True)
    test = df[df["Date"] >= split_date].reset_index(drop=True)

    # Buy at first test day
    first_price = test.iloc[0]["Close"]
    shares = initial_cash / first_price

    # Portfolio value over time
    test["Portfolio"] = shares * test["Close"]

    # Metrics
    final_value = test["Portfolio"].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash
    daily_returns = test["Portfolio"].pct_change().dropna()
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    print(f"Initial cash: {initial_cash:.2f}")
    print(f"Final portfolio value: {final_value:.2f}")
    print(f"Total return: {total_return*100:.2f}%")
    print(f"Sharpe ratio: {sharpe:.2f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(test["Date"], test["Portfolio"], label="Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Buy & Hold Baseline Strategy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    buy_and_hold("data/AAPL.csv")
