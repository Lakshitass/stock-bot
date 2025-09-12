# src/eval.py
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from src.env import TradingEnv

def evaluate(model_path, data_path, plot_path="results/plots/equity_curve.png"):
    # create a fresh (non-vectorized) environment for evaluation
    env = TradingEnv(data_path)
    model = PPO.load(model_path)  # if saved as results/models/ppo_aapl.zip -> pass that base path

    obs = env.reset()
    net_worths = [env.net_worth]
    dates = [env.df.loc[env.current_step, "Date"]]
    prices = [env.df.loc[env.current_step, "Close"]]
    actions_list = []

    done = False
    while not done:
        # deterministic=True often gives stable evaluation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        net_worths.append(env.net_worth)
        # guard in case current_step hits end
        if env.current_step < len(env.df):
            dates.append(env.df.loc[env.current_step, "Date"])
            prices.append(env.df.loc[env.current_step, "Close"])
        actions_list.append(int(action))

    # ensure output dir exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    plt.figure(figsize=(12,6))
    plt.plot(net_worths, label="RL Net Worth")
    plt.title("RL Agent Equity Curve")
    plt.xlabel("Steps")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()
    print("Evaluation done, plot saved to", plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/models/ppo_aapl.zip")
    parser.add_argument("--env", type=str, default="data/AAPL_test.csv")
    parser.add_argument("--out", type=str, default="results/plots/equity_curve.png")
    args = parser.parse_args()
    evaluate(args.model, args.env, args.out)
