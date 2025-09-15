# evaluate_and_plot.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, DQN
from env_continuous import ContinuousAllocationEnv
from env_discrete import DiscreteTradingEnv

def make_env_from_csv(csv_path, continuous=True, window_size=10, cash=10000, seed=0):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if continuous:
        return ContinuousAllocationEnv(price_df=df, window_size=window_size, initial_cash=cash, seed=seed)
    else:
        return DiscreteTradingEnv(price_df=df, window_size=window_size, initial_cash=cash, seed=seed)

def evaluate_model(model, env):
    # reset handling (gym vs gymnasium)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out

    equity = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        # support both 4-tuple and 5-tuple step returns
        if len(step_out) == 4:
            obs, reward, done, info = step_out
        else:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated

        # read portfolio value from info or env
        pv = info.get("portfolio_value", getattr(env, "portfolio_value", None))
        equity.append(float(pv) if pv is not None else np.nan)
        if done:
            break
    return equity

def plot_equity(equity, out_path, title="Equity Curve"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(10,5))
    plt.plot(equity, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved equity plot to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--algo", required=True, choices=["sac","dqn"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results/equity.png")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--cash", type=float, default=10000)
    args = parser.parse_args()

    continuous = args.algo == "sac"
    env = make_env_from_csv(args.data, continuous=continuous, window_size=args.window, cash=args.cash)
    if args.algo == "sac":
        model = SAC.load(args.model)
    else:
        model = DQN.load(args.model)

    equity = evaluate_model(model, env)
    plot_equity(equity, args.out, title=f"{args.algo.upper()} Equity Curve")
