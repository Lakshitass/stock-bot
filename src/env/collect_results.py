# collect_results.py
import os
import csv
import numpy as np
from evaluate_and_plot import evaluate_model, make_env_from_csv
from stable_baselines3 import SAC, DQN

def metrics_from_equity(equity, initial_cash):
    equity = np.array(equity)
    final = equity[-1]
    total_return = (final - initial_cash) / initial_cash * 100.0
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    max_drawdown = np.nanmax(drawdowns) * 100.0
    return final, total_return, max_drawdown

def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model","algo","stock","timesteps","initial_cash","final_pv","total_return_pct","max_drawdown_pct"])
        writer.writerow(row)

if __name__ == "__main__":
    initial_cash = 10000
    data = "data/AAPL.csv"
    models = [
        ("models/sac_aapl.zip", "sac", 5000),
        ("models/dqn_aapl.zip", "dqn", 5000),
    ]

    for model_path, algo, timesteps in models:
        if algo == "sac":
            model = SAC.load(model_path)
            env = make_env_from_csv(data, continuous=True, window_size=10, cash=initial_cash)
        else:
            model = DQN.load(model_path)
            env = make_env_from_csv(data, continuous=False, window_size=10, cash=initial_cash)

        equity = evaluate_model(model, env)
        final, tot_ret, max_dd = metrics_from_equity(equity, initial_cash)
        append_row("results/summary.csv", [model_path, algo, "AAPL", timesteps, initial_cash, final, tot_ret, max_dd])
        print(f"Appended results for {model_path}")
