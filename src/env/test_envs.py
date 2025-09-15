# test_envs.py
"""
Quick test runner for the DiscreteTradingEnv and ContinuousAllocationEnv.
Runs a random policy for a few steps and prints outputs.
"""

import numpy as np
import pandas as pd
from env_discrete import DiscreteTradingEnv
from env_continuous import ContinuousAllocationEnv

def make_dummy_price_df(n_steps=200, n_assets=1, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_steps, freq="B")
    # generate geometric brownian-like prices
    df = pd.DataFrame(index=dates)
    for i in range(n_assets):
        price = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n_steps)))
        df[f"T{i}" if n_assets>1 else "AAPL"] = price
    return df

def run_discrete_test():
    print("=== Discrete Env Test ===")
    price_df = make_dummy_price_df(n_steps=120, n_assets=1)
    env = DiscreteTradingEnv(price_df=price_df, window_size=5, initial_cash=10000, transaction_cost_pct=0.001, seed=0)
    obs, _ = env.reset(seed=0)
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < 30:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        env.render()
        steps += 1
    print("Total reward (PV change):", total_reward)
    env.close()

def run_continuous_test():
    print("=== Continuous Env Test ===")
    price_df = make_dummy_price_df(n_steps=120, n_assets=2)
    env = ContinuousAllocationEnv(price_df=price_df, window_size=5, initial_cash=50000, transaction_cost_pct=0.001, seed=1)
    obs, _ = env.reset(seed=1)
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < 30:
        action = env.action_space.sample()
        # normalize action so agents produce allocation (sums to <=1)
        if action.sum() > 1.0:
            action = action / action.sum()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        env.render()
        steps += 1
    print("Total reward (PV change):", total_reward)
    env.close()

if __name__ == "__main__":
    run_discrete_test()
    run_continuous_test()
