from stable_baselines3 import PPO
from src.env import TradingEnv
import matplotlib.pyplot as plt

env = TradingEnv("data/AAPL.csv")
model = PPO.load("results/models/ppo_aapl")

obs = env.reset()
net_worths = []

for _ in range(len(env.df)-1):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    net_worths.append(env.net_worth)
    if done:
        break

plt.plot(net_worths)
plt.title("Agent Net Worth Over Time")
plt.xlabel("Steps")
plt.ylabel("Net Worth")
plt.savefig("results/plots/equity_curve.png")
plt.show()

