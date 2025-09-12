# src/train.py
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.env import TradingEnv

def train(env_path, timesteps=20000, save_path="results/models/ppo_aapl"):
    # create a vectorized environment factory
    env = make_vec_env(lambda: TradingEnv(env_path), n_envs=1)

    # create PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # train
    model.learn(total_timesteps=timesteps)

    # save model (SB3 will add .zip automatically when using save)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="data/AAPL_train.csv")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--save", type=str, default="results/models/ppo_aapl")
    args = parser.parse_args()

    train(args.env, args.timesteps, args.save)
