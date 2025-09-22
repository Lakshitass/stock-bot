# train_dqn.py
import os
import argparse
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env_discrete import DiscreteTradingEnv

def make_env(csv_path, window_size, initial_cash, seed=0):
    def _init():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        env = DiscreteTradingEnv(price_df=df, window_size=window_size, initial_cash=initial_cash, seed=seed)
        return Monitor(env)
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--out", default="models/dqn_aapl")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--cash", type=float, default=10000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # DQN: use single env (or n_envs=1)
    vec_env = DummyVecEnv([make_env(args.data, args.window, args.cash, seed=0)])

    model = DQN("MlpPolicy", vec_env, verbose=1,
                learning_rate=1e-4,
                buffer_size=50000,
                batch_size=32,
                gamma=0.99,
                exploration_fraction=0.1,
                exploration_final_eps=0.02)

    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print("Saved DQN model to", args.out)

if __name__ == "__main__":
    main()
