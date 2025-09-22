# env_continuous.py
import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnvContinuous(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_balance=10000):
        super(TradingEnvContinuous, self).__init__()

        self.data = data.reset_index(drop=True)
        self.n_steps = len(data)
        self.initial_balance = initial_balance

        # action = fraction of balance to allocate [-1, 1] (sell short to full buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        return self._get_obs()

    def _get_obs(self):
        price = self.data.iloc[self.current_step]["Close"]
        return np.array([
            self.balance,
            self.shares_held,
            price,
            self.net_worth,
            self.max_net_worth
        ], dtype=np.float32)

    def step(self, action):
        price = self.data.iloc[self.current_step]["Close"]
        fraction = float(action[0])  # scale [-1, 1]

        # fraction < 0 → sell, fraction > 0 → buy
        if fraction > 0:  # buy
            amount_to_spend = fraction * self.balance
            shares_bought = amount_to_spend // price
            self.balance -= shares_bought * price
            self.shares_held += shares_bought
        elif fraction < 0:  # sell
            shares_to_sell = int(abs(fraction) * self.shares_held)
            self.balance += shares_to_sell * price
            self.shares_held -= shares_to_sell

        # update
        self.current_step += 1
        self.net_worth = self.balance + self.shares_held * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        reward = self.net_worth - self.initial_balance
        done = self.current_step >= self.n_steps - 1
        return self._get_obs(), reward, done, {}
