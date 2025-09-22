# batch_run_all.py
"""
Batch-run predictions and optional DQN training for every CSV in ./data/
Outputs for each stock are written to ./results/<stock_name>/ and
also a combined comparison plot for all stocks.

Usage:
    python batch_run_all.py --data-dir ./data --out-dir ./results --mode direct
    python batch_run_all.py --data-dir ./data --out-dir ./results --mode subprocess
"""

import os
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from tqdm import tqdm
import json

# ------------------------------
# Helper: Add simple indicators
# ------------------------------
def add_basic_indicators(df):
    df = df.copy()
    df['close'] = df['Close'].astype(float)
    df['return'] = df['close'].pct_change()
    df['vol_10'] = df['return'].rolling(10).std()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_slope'] = df['sma_10'] - df['sma_50']
    df['momentum_5'] = df['close'].pct_change(5)
    df['atr_14'] = df['High'].astype(float).combine(df['Low'].astype(float), max) - \
                   df['High'].astype(float).combine(df['Low'].astype(float), min)
    df = df.fillna(method='bfill').fillna(0)
    return df

# ------------------------------
# Predictor dataset
# ------------------------------
def prepare_predictor_dataset(df):
    df = add_basic_indicators(df)
    feature_cols = ['close','sma_10','sma_50','sma_slope','momentum_5','vol_10','atr_14']
    X = df[feature_cols].shift(0).fillna(method='bfill')
    y = df['return'].shift(-1).fillna(0)
    X, y = X.iloc[:-1], y.iloc[:-1]
    return X, y, df

# ------------------------------
# Train RandomForest
# ------------------------------
def train_predictor(X, y, model_path=None, force=False):
    if model_path and os.path.exists(model_path) and not force:
        return joblib.load(model_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Predictor trained, test MSE={mse:.6f}")
    if model_path:
        joblib.dump(model, model_path)
    return model

# ------------------------------
# Direct run for one stock
# ------------------------------
def run_for_stock(csv_path, out_dir, force_predictor=False):
    stock_name = Path(csv_path).stem
    stock_out = Path(out_dir) / stock_name
    stock_out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Processing {stock_name} ===")

    df = pd.read_csv(csv_path)
    if "Close" not in df.columns:
        print(f"ERROR: {csv_path} missing Close column.")
        return None

    X, y, df_feat = prepare_predictor_dataset(df)
    model_path = stock_out / f"{stock_name}_rf.pkl"
    rf = train_predictor(X, y, str(model_path), force=force_predictor)

    preds_all = rf.predict(X)
    df_pred = df_feat.iloc[:-1].copy()
    df_pred["pred_next_return"] = preds_all
    df_pred["pred_next_price"] = df_pred["close"] * (1 + df_pred["pred_next_return"])
    actual_next_return = df_feat['close'].shift(-1).iloc[:-1] / df_pred['close'] - 1

    # Compute evaluation metrics
    mse = mean_squared_error(actual_next_return, df_pred["pred_next_return"])
    directional_acc = np.mean(np.sign(df_pred["pred_next_return"]) == np.sign(actual_next_return)) * 100

    pred_csv = stock_out / f"{stock_name}_predictions.csv"
    df_pred.to_csv(pred_csv, index=False)
    print(f"Saved predictions: {pred_csv}")
    print(f"{stock_name} MSE={mse:.6f}, Directional Accuracy={directional_acc:.2f}%")

    # Individual stock plot
    try:
        plt.figure()
        plt.plot(df_pred["close"].values, label="Actual Price", linestyle='--', alpha=0.7)
        plt.plot(df_pred["pred_next_price"].values, label="Predicted Next Price", linewidth=1.5)
        plt.legend()
        plt.title(f"{stock_name} Prediction")
        plot_path = stock_out / f"{stock_name}_plot.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    metrics = {
        "stock": stock_name,
        "pred_csv": str(pred_csv),
        "mse": mse,
        "directional_accuracy": directional_acc
    }
    with open(stock_out / f"{stock_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

# ------------------------------
# Main function
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./results")
    parser.add_argument("--mode", choices=["direct", "subprocess"], default="direct")
    parser.add_argument("--force-retrain-predictor", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in", data_dir)
        return

    all_metrics = []
    stock_names = []

    if args.mode == "direct":
        for csv in tqdm(csv_files, desc="Stocks"):
            metrics = run_for_stock(csv, out_dir, force_predictor=args.force_retrain_predictor)
            if metrics:
                all_metrics.append(metrics)
                stock_names.append(metrics["stock"])

    elif args.mode == "subprocess":
        for csv in tqdm(csv_files, desc="Stocks"):
            stock_name = csv.stem
            stock_out = out_dir / stock_name
            stock_out.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "python", "dqn_stock_agent.py",
                "--data-path", str(csv),
                "--out-dir", str(stock_out)
            ])
            stock_names.append(stock_name)

    # Save summary CSV
    if all_metrics:
        df_sum = pd.DataFrame(all_metrics)
        df_sum.to_csv(out_dir / "summary.csv", index=False)
        print("Saved summary:", out_dir / "summary.csv")

    # ------------------------------
    # Combined plot for all stocks
    # ------------------------------
    plt.figure(figsize=(12,6))
    colors = plt.cm.tab10.colors  # 10 distinct colors
    for i, stock in enumerate(stock_names):
        pred_csv = out_dir / stock / f"{stock}_predictions.csv"
        if pred_csv.exists():
            df = pd.read_csv(pred_csv)
            plt.plot(df['pred_next_price'], label=f"{stock} predicted", color=colors[i % 10], linewidth=1.5)
            plt.plot(df['close'], label=f"{stock} actual", linestyle='--', alpha=0.6, color=colors[i % 10])
        else:
            print(f"CSV not found for {stock}: {pred_csv}")

    plt.title("Predicted vs Actual Prices for All Stocks")
    plt.xlabel("Days (CSV index)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    combined_plot_path = out_dir / "all_stocks_comparison.png"
    plt.savefig(combined_plot_path, bbox_inches="tight")
    print(f"Saved combined plot: {combined_plot_path}")

    # Automatically open the combined plot on Windows
    if os.name == "nt" and combined_plot_path.exists():
        os.startfile(combined_plot_path)

if __name__ == "__main__":
    main()
