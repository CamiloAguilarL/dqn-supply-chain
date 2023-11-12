import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
import math


class SupplyChainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(self, render_mode=None, render_fps=240, num_periods=10, num_products=1, num_suppliers=1, max_demand=1000, min_price=1, max_price=50, max_quantity=2000, perishability=0.3, buy_bins=10, sell_min_price=5, sell_max_price=200):
        self.num_periods = num_periods
        self.num_products = num_products
        self.num_suppliers = num_suppliers
        self.max_demand = max_demand
        self.min_price = min_price
        self.max_price = max_price
        self.max_quantity = max_quantity
        self.buy_bins = buy_bins
        self.perishability = perishability
        self.sell_min_price = sell_min_price
        self.sell_max_price = sell_max_price
        self.sell_price = self.np_random.uniform(low=sell_min_price, high=sell_max_price, size=self.num_products)
        self.period = 0
        self.window_size = 512  # The size of the PyGame window

        self.action_space = spaces.Box(low=0, high=buy_bins, shape=(num_suppliers, num_products), dtype=np.int64)

        self.observation_space = spaces.Dict({
            "inventory": spaces.Box(low=0, high=np.inf, shape=(num_products,), dtype=np.int64),
            "demand": spaces.Box(low=0, high=max_demand, shape=(num_products,), dtype=np.int64),
            "price": spaces.Box(low=min_price, high=max_price, shape=(num_suppliers, num_products), dtype=np.int64),
            "quantity": spaces.Box(low=0, high=max_quantity, shape=(num_suppliers, num_products), dtype=np.int64),
        })

        # Initial state
        self.state = self.reset()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.metadata['render_fps'] = render_fps
        
        self.window = None
        self.clock = None

    def step(self, action):
        # Decompose action into purchasing and routing (simplified for this example)
        purchasing_decision = action  # This is a simplification
        # Update inventory based on purchasing decision and existing inventory
        purchased_inventory = np.sum(self.state['quantity'] * purchasing_decision/self.buy_bins, axis=0, dtype=int)  # Total purchased per product

        cost = np.sum(purchased_inventory * self.state['price'])

        self.state['inventory'] += purchased_inventory

        # Calculate sales and update inventory based on demand
        sales = np.minimum(self.state['demand'], self.state['inventory'])
        self.state['inventory'] -= sales

        self.state['inventory'] = np.floor(self.state['inventory']*(1-self.perishability))
        # Calculate revenue from sales
        revenue = np.sum(sales * self.sell_price)

        # Calculate penalty for not meeting demand
        penalty = np.sum((self.state['demand'] - sales) * 10)
        penalty = max(0, penalty)  # Ensure penalty is not negative

        # Calculate reward (revenue minus penalty)
        reward = revenue - cost - penalty

        # Update state randomly (for demand, price, and quantity)
        self.state['demand'] = self.np_random.integers(low=0, high=self.max_demand, size=self.num_products)
        self.state['price'] = self.np_random.integers(low=self.min_price, high=self.max_price, size=(self.num_suppliers, self.num_products))
        self.state['quantity'] = self.np_random.integers(low=0, high=self.max_quantity, size=(self.num_suppliers, self.num_products))

        # Check if the episode is done (here we check if we reached the last period)
        self.period += 1
        done = self.period >= self.num_periods

        info = {}  # Additional info, if any

        return self.state, reward, done, False, info
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the environment to an initial state
        initial_state = {
            "inventory": np.zeros(self.num_products, dtype=np.int64),
            "demand": self.np_random.integers(low=0, high=self.max_demand, size=self.num_products),
            "price": self.np_random.integers(low=self.min_price, high=self.max_price, size=(self.num_suppliers, self.num_products)),
            "quantity": self.np_random.integers(low=0, high=self.max_quantity, size=(self.num_suppliers, self.num_products)),
        }
        self.period = 0
        self.state = initial_state
        self.info = {}
        return self.state, self.info
