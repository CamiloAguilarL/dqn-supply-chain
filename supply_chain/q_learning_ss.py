import numpy as np
import itertools
import copy

from agent_interface import AgentInterface

class QLearningSSAgent(AgentInterface):
    def __init__(self, env, bins=5, learning_rate=0.2, discount_factor=0.9, exploration_prob=0.3, min_exploration_prob=0.01, decay_rate=0.00005):
        self.env = env
        self.observation_space = env.unwrapped.observation_space
        self.action_space = env.unwrapped.action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.decay_rate = decay_rate
        self.bins = bins
        self.inventory_intervals = np.linspace(0, env.unwrapped.max_demand, num=bins)
        self.demand_intervals = np.linspace(env.unwrapped.min_demand, env.unwrapped.max_demand, num=bins)
        self.price_intervals = np.linspace(env.unwrapped.min_price, env.unwrapped.max_price, num=bins)
        self.quantity_intervals = np.linspace(env.unwrapped.min_quantity, env.unwrapped.max_quantity, num=bins)

        # Initialize Q-table as a dictionary
        self.q_table = {}

    def _get_state_key(self, state):
        inventory_level = tuple(np.digitize(state["inventory"], self.inventory_intervals))
        demand_level = tuple(np.digitize(state["demand"], self.demand_intervals))
        price_level = tuple([tuple(np.digitize(price, self.price_intervals)) for price in state["price"]])
        # quantity_level = tuple([tuple(np.digitize(quantity, self.quantity_intervals)) for quantity in state["quantity"]])
        return inventory_level, demand_level, price_level
    
    def _get_action_key(self, action):
        products_sum = np.sum(action, axis=0)
        return tuple((i, 1 if j > 0 else 0,) for i, j in zip(range(self.env.unwrapped.num_products),products_sum))
    
    def _get_action_space(self):
        combinations = range(self.env.unwrapped.num_products)
        buy_combinations = list(itertools.product([0,1], repeat=self.env.unwrapped.num_products))
        space = []
        for b in buy_combinations:
            action_list = list(zip(combinations, b))
            action_items = ((i, j) for i, j in action_list)
            space.append(tuple(action_items))
        return space

    def _get_action_value(self, state, action_key):
        replenish_quantity = 2*self.env.unwrapped.max_demand
        price = state['price']
        quantity = state['quantity']
        action = np.zeros((self.env.unwrapped.num_suppliers, self.env.unwrapped.num_products), dtype=int)
        for i, buy in action_key:
            if buy == 1:
                visited_suppliers = []
                total_quantity = 0
                while len(visited_suppliers) < self.env.unwrapped.num_suppliers:
                    min_price = price[0][i]
                    min_price_index = 0
                    for j, supplier in enumerate(price):
                        if supplier[i] < min_price and j not in visited_suppliers:
                            min_price = supplier[i]
                            min_price_index = j
                    visited_suppliers.append(min_price_index)
                    buy_quantity = min(replenish_quantity - total_quantity, quantity[min_price_index, i])
                    total_quantity += buy_quantity
                    action[j,i] = buy_quantity
                    if total_quantity > replenish_quantity:
                        break
        return action

    def choose_action(self, state, greedy=False):
        state_key = self._get_state_key(state)
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_prob and not greedy:
            if state_key not in self.q_table:
                self.q_table[state_key] = {action_key: 0 for action_key in self._get_action_space()}
            random_index = np.random.randint(0, len(self.q_table[state_key].keys()))
            action_key = list(self.q_table[state_key].keys())[random_index]
            value = self._get_action_value(state, action_key)
            return value
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = {action_key: 0 for action_key in self._get_action_space()}
            action_key = max(self.q_table[state_key], key=self.q_table[state_key].get)
            return self._get_action_value(state, action_key)

    def update(self, state, action, next_state, reward):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        action_key = self._get_action_key(action)

        if state_key not in self.q_table:
            self.q_table[state_key] = {action_key: 0 for action_key in self._get_action_space()}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {action_key: 0 for action_key in self._get_action_space()}

        max_action_key = max(self.q_table[next_state_key], key=self.q_table[next_state_key].get)
        max_action = 0 if next_state_key not in self.q_table else self.q_table[next_state_key][max_action_key]
        # Q-value update using the Q-learning formula
        self.q_table[state_key][action_key] = (
            1 - self.learning_rate
        ) * self.q_table[state_key][action_key] + self.learning_rate * (
            reward + self.discount_factor * max_action
        )

    def train(self, num_episodes=1000, load=False, save=False, train=True, filename="q_table.npy", seed=None):
        if load:
            self.q_table = np.load(filename, allow_pickle=True).item()
        if train:
            rewards = []
            for _ in range(num_episodes):
                state, info = self.env.unwrapped.reset(seed=seed)
                score = 0
                while True:
                    action = self.choose_action(state)
                    prev_state = copy.deepcopy(state)
                    next_state, reward, terminated, truncated, info = self.env.unwrapped.step(action)
                    self.update(prev_state, action, next_state, reward)
                    state = next_state
                    score += reward
                    if terminated or truncated:
                        break
                rewards.append(score/self.env.unwrapped.num_periods)
                self.exploration_prob = max(self.exploration_prob * (1-self.decay_rate), self.min_exploration_prob)
                if _ % 100 == 0:
                    print(f"Training Agent: Q Learning SS, Episode: {_}, Score: {score}")
            if save:
                np.save(filename, self.q_table)

            return rewards