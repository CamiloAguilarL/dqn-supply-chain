import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
import math

gym.logger.set_level(40)

class SupplyChainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(self, 
                 action_mode="discrete",
                 render_mode=None, 
                 render_fps=240, 
                 num_periods=20, 
                 num_products=5,
                 num_suppliers=20,
                 min_demand=1000,
                 max_demand=1500, 
                 sell_price_percentage=0.1, 
                 min_quantity=400,
                 max_quantity=900, 
                 perishability=0.1,
                 buy_bins=3, 
                 min_price=0.5, 
                 max_price=1.2,
                 min_transport_time=500,
                 max_transport_time=600,
                 unit_transport_cost=1,
                 backorder_price_percentage=250,
                 vehicle_capacity=60,
                 loading_time=10,
                 travel_speed=60,
                 region_size = 250):
        
        self.action_mode = action_mode
        self.num_periods = num_periods
        self.num_products = num_products
        self.num_suppliers = num_suppliers
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.sell_price_percentage = sell_price_percentage
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.buy_bins = buy_bins
        self.perishability = perishability
        self.min_price = min_price
        self.max_price = max_price
        self.min_transport_time = min_transport_time
        self.max_transport_time = max_transport_time
        self.unit_transport_cost = unit_transport_cost
        self.backorder_price_percentage = backorder_price_percentage
        self.vehicle_capacity = vehicle_capacity
        self.loading_time = loading_time
        self.travel_speed = travel_speed
        self.region_size = region_size
        self.period = 0
        self.window_size = 512  # The size of the PyGame window

        if action_mode == "discrete":
            self.action_space = spaces.Box(low=0, high=buy_bins, shape=(num_suppliers, num_products), dtype=np.int64)
        else: 
            self.action_space = spaces.Box(low=0, high=np.inf, shape=(num_suppliers, num_products), dtype=np.int64)

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

        if self.action_mode == "discrete":
            # Update inventory based on purchasing decision and existing inventory
            purchased_quantity = self.state['quantity'] * purchasing_decision/self.buy_bins
            purchased_inventory = np.sum(purchased_quantity, axis=0, dtype=int)  # Total purchased per product
        else:
            purchased_quantity = purchasing_decision
            purchased_inventory = np.sum(purchasing_decision, axis=0, dtype=int)

        suppliers_visited = np.where(np.sum(purchasing_decision, axis=1) > 0, 1, 0)

        self.state['inventory'] = np.floor(self.state['inventory']*(1-self.perishability))

        cost = np.sum(purchased_quantity * self.state['price']) + self.unit_transport_cost * np.sum(suppliers_visited * self.transport_time)
        self.state['inventory'] += purchased_inventory

        # Calculate sales and update inventory based on demand
        sales = np.minimum(self.state['demand'], self.state['inventory'])
        self.state['inventory'] -= sales

        # Calculate revenue from sales
        revenue = np.sum(sales * self.sell_price)

        # Calculate penalty for not meeting demand
        penalty = np.sum((self.state['demand'] - sales) * self.backorder_price)
        penalty = max(0, penalty)  # Ensure penalty is not negative

        # Calculate reward (revenue minus penalty)
        reward = revenue - cost - penalty

        # Update state randomly (for demand, price, and quantity)
        self.state['demand'] = np.clip(self.np_random.normal(self.mean_demand, 0.1, self.num_products), 0, 2*self.mean_demand)
        self.state['price'] = np.where(self.products_suppliers == 1, np.clip(self.np_random.normal(self.mean_price, 0.1, (self.num_suppliers, self.num_products)), 1, 2*self.mean_price - 1), 10*self.mean_price)
        self.state['quantity'] = np.where(self.products_suppliers == 1, np.clip(self.np_random.normal(self.mean_quantity, 0.1, (self.num_suppliers, self.num_products)), 0, 2*self.mean_quantity), 0)
        self.sell_price = np.mean(self.mean_price, axis=0) * (1 + self.sell_price_percentage)
        self.backorder_price = self.sell_price * self.backorder_price_percentage

        # Check if the episode is done (here we check if we reached the last period)
        self.period += 1
        done = self.period >= self.num_periods
        info = {}  # Additional info, if any
        return self.state, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        initial_state = {}
        products_suppliers = np.zeros((self.num_suppliers, self.num_products), dtype=np.int64)
        transport_time = np.zeros(self.num_suppliers, dtype=np.int64)
        for i in range(self.num_suppliers):
            for j in range(self.num_products):
                products_suppliers[i][j] = 0 if self.np_random.uniform() > 0.3 else 1
            if products_suppliers[i].sum() == 0:
                products_suppliers[i][self.np_random.integers(0, self.num_products)] = 1
            coordinates = (self.np_random.uniform(-self.region_size/2, self.region_size/2), self.np_random.uniform(-self.region_size/2, self.region_size/2))
            transport_time[i] = (math.sqrt(coordinates[0]**2 + coordinates[1]**2)/self.travel_speed)*60 + self.loading_time

        for i in range(self.num_products):
            if products_suppliers[:,i].sum() == 0:
                products_suppliers[self.np_random.integers(0, self.num_suppliers)][i] = 1

        self.transport_time = transport_time
        self.products_suppliers = products_suppliers

        self.mean_demand = self.np_random.uniform(low=self.min_demand, high=self.max_demand, size=self.num_products)
        self.mean_quantity = self.np_random.uniform(low=self.min_quantity, high=self.max_quantity, size=(self.num_suppliers, self.num_products))
        self.mean_price = self.np_random.uniform(low=self.min_price, high=self.max_price, size=(self.num_suppliers, self.num_products))

        initial_state = {
            "inventory": np.zeros(self.num_products, dtype=np.int64),
            "demand": np.zeros(self.num_products, dtype=np.int64),
            "price": np.where(self.products_suppliers == 1, np.clip(self.np_random.normal(self.mean_price, 0.1, (self.num_suppliers, self.num_products)), 1, 2*self.mean_price - 1), 10*self.mean_price),
            "quantity": np.where(self.products_suppliers == 1, np.clip(self.np_random.normal(self.mean_quantity, 0.1, (self.num_suppliers, self.num_products)), 0, 2*self.mean_quantity), 0)
        }
        self.sell_price = np.mean(self.mean_price, axis=0) * (1 + self.sell_price_percentage)
        self.backorder_price = self.sell_price * self.backorder_price_percentage

        self.period = 0
        self.state = initial_state
        self.info = {}
        return self.state, self.info
