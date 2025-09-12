from stable_baselines3 import PPO
from src.env import TradingEnv

def main():
    env = TradingEnv("data/AAPL.csv")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("results/models/ppo_aapl")

if __name__ == "__main__":
    main()
