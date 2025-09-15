# env_discrete.py
"""
Discrete trading environment (Buy / Hold / Sell).
Supports single stock (Discrete(3)) or multiple stocks (MultiDiscrete([3]*n)).
Observation: last N prices normalized + current cash + current holdings.
Action mapping per stock:
    0 -> SELL (sell all shares of that stock)
    1 -> HOLD
    2 -> BUY (buy as many shares as possible with equal split of available cash)
Reward: change in total portfolio value (PV_t - PV_{t-1})
"""

from typing import Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class DiscreteTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        price_df: Union[pd.Series, pd.DataFrame],
        window_size: int = 10,
        initial_cash: float = 100000.0,
        max_shares_per_trade: Optional[int] = None,
        transaction_cost_pct: float = 0.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._rng = np.random.default_rng(seed)

        # Normalize price_df into DataFrame with columns = tickers (for a single series it becomes one column)
        if isinstance(price_df, pd.Series):
            price_df = price_df.to_frame("PRICE")
        self.price_df = price_df.reset_index(drop=True)
        self.dates = self.price_df.index.astype(int).to_numpy()
        self.tickers = list(self.price_df.columns)
        self.n_assets = len(self.tickers)

        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.max_shares_per_trade = max_shares_per_trade

        # action_space: for single stock Discrete(3), for multi-stock MultiDiscrete([3]*n)
        if self.n_assets == 1:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.MultiDiscrete([3] * self.n_assets)

        # observation: window_size normalized prices for each asset + cash + holdings vector
        # price window: (n_assets * window_size)
        obs_low = -np.inf * np.ones(self.n_assets * self.window_size + 1 + self.n_assets)
        obs_high = np.inf * np.ones_like(obs_low)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.render_mode = render_mode

        # internal state
        self.reset()

    def reset(self, seed: Optional[int] = None, options: dict = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.current_step = self.window_size - 1  # index into price_df
        self.cash = float(self.initial_cash)
        self.holdings = np.zeros(self.n_assets, dtype=np.int64)  # integer shares
        self.done = False
        self._update_portfolio_value()
        self.last_portfolio_value = self.portfolio_value
        return self._get_obs(), {}

    def _get_price(self, step_index: int):
        return self.price_df.iloc[step_index].to_numpy(dtype=float)

    def _get_obs(self):
        # price window (window_size x n_assets) flattened, normalized by first price in window
        start = self.current_step - (self.window_size - 1)
        window = self.price_df.iloc[start : self.current_step + 1].to_numpy(dtype=float)  # shape (window, n_assets)
        # normalize each asset by its first price in the window to reduce scale
        norm = window / (window[0] + 1e-9)
        obs = np.concatenate([norm.flatten(), np.array([self.cash], dtype=float), self.holdings.astype(float)])
        return obs.astype(np.float32)

    def _update_portfolio_value(self):
        price = self._get_price(self.current_step)
        self.portfolio_value = self.cash + np.dot(self.holdings, price)

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called on done environment; call reset().")

        # unify action to array per asset
        if self.n_assets == 1:
            action_arr = np.array([int(action)])
        else:
            action_arr = np.array(action, dtype=int)

        price = self._get_price(self.current_step)

        # Execute actions for each asset
        for i, act in enumerate(action_arr):
            if act == 0:  # SELL: sell all shares of asset i
                shares_to_sell = self.holdings[i]
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price[i]
                    cost = proceeds * self.transaction_cost_pct
                    self.cash += proceeds - cost
                    self.holdings[i] = 0
            elif act == 2:  # BUY: buy as many shares as possible but optionally limited
                # Strategy: split available cash equally among assets that want to buy
                # We'll perform a naive immediate buy per asset using equal share of cash among BUY requests
                pass  # we'll handle buys after counting buyers

        # handle buys: compute how many assets wanted to buy
        buyers = np.where(action_arr == 2)[0]
        if buyers.size > 0:
            cash_per_buyer = self.cash / buyers.size
            for i in buyers:
                # how many shares we can buy
                shares = int(cash_per_buyer // price[i])
                if self.max_shares_per_trade is not None:
                    shares = min(shares, self.max_shares_per_trade)
                if shares <= 0:
                    continue
                spend = shares * price[i]
                cost = spend * self.transaction_cost_pct
                total_spend = spend + cost
                # if rounding issues overspend, skip or adjust
                if total_spend > self.cash + 1e-8:
                    continue
                self.cash -= total_spend
                self.holdings[i] += shares

        # Advance step
        prev_portfolio = getattr(self, "portfolio_value", None)
        self._update_portfolio_value()
        reward = self.portfolio_value - (prev_portfolio if prev_portfolio is not None else self.initial_cash)

        # increment day
        self.current_step += 1
        if self.current_step >= len(self.price_df):
            self.done = True
            # final step reward included already
        info = {
            "step": self.current_step,
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.cash),
            "holdings": self.holdings.copy(),
        }

        obs = self._get_obs() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), bool(self.done), False, info

    def render(self):
        price = self._get_price(min(self.current_step, len(self.price_df) - 1))
        print(
            f"Step: {self.current_step} | PV: {self.portfolio_value:.2f} | Cash: {self.cash:.2f} | Holdings: {self.holdings} | Prices: {price}"
        )

    def close(self):
        return
