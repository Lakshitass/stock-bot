# train_sac.py
import os
import argparse
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env_continuous import ContinuousAllocationEnv

def make_env(csv_path, window_size, initial_cash, seed=0):
    def _init():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        env = ContinuousAllocationEnv(price_df=df, window_size=window_size, initial_cash=initial_cash, seed=seed)
        return Monitor(env)
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--out", default="models/sac_aapl")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--cash", type=float, default=10000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # create vectorized envs
    env_fns = [make_env(args.data, args.window, args.cash, seed=i) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Small example hyperparams (tune later)
    model = SAC("MlpPolicy", vec_env, verbose=1,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                gamma=0.99)

    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print("Saved model to", args.out)

if __name__ == "__main__":
    main()
