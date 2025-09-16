"""
evaluate.py – prints performance metrics for each stock
Place inside: stock-bot/src/
Run with:  python src/evaluate.py
"""

import pandas as pd
from pathlib import Path
from metrics import total_return, annualized_return, sharpe_ratio, max_drawdown

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STOCKS   = ["AAPL", "TSLA", "JPM", "GS", "XOM"]

def evaluate_stock(csv_path: Path):
    # Load price data
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    # For now treat 'Close' price as equity curve
    equity = df.set_index("Date")["Close"]
    daily_ret = equity.pct_change().dropna()

    return {
        "Total Return"     : f"{total_return(equity):.2%}",
        "Annualized Return": f"{annualized_return(equity):.2%}",
        "Sharpe Ratio"     : f"{sharpe_ratio(daily_ret):.2f}",
        "Max Drawdown"     : f"{max_drawdown(equity):.2%}"
    }

def main():
    print("=== Evaluation of Close Prices ===")
    for stock in STOCKS:
        csv_file = DATA_DIR / f"{stock}.csv"
        if not csv_file.exists():
            print(f"{stock}: CSV not found in data/")
            continue
        metrics = evaluate_stock(csv_file)
        print(f"\n{stock}")
        for k, v in metrics.items():
            print(f"  {k:<18}: {v}")

if __name__ == "__main__":
    main()
