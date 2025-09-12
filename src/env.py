import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data_path):
        super(TradingEnv, self).__init__()
        self.df = pd.read_csv(data_path)
        self.current_step = 0

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observations: just price for now
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        self.balance = 10000
        self.shares_held = 0
        self.net_worth = 10000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = 10000
        obs = np.array([self.df.loc[self.current_step, "Close"]], dtype=np.float32)
        return obs, {}

    def step(self, action):
        price = self.df.loc[self.current_step, "Close"]

        if action == 1:  # Buy
            self.shares_held += 1
            self.balance -= price
        elif action == 2 and self.shares_held > 0:  # Sell
            self.shares_held -= 1
            self.balance += price

        self.current_step += 1
        self.net_worth = self.balance + self.shares_held * price

        reward = self.net_worth - 10000  # profit relative to start
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        obs = np.array([self.df.loc[self.current_step, "Close"]], dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth}")
