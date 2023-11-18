import numpy as np
from agent_interface import AgentInterface
import itertools

class QLearningAgent(AgentInterface):
    def __init__(self, env, bins=5, learning_rate=0.2, discount_factor=0.9, exploration_prob=0.2):
        self.env = env
        self.observation_space = env.unwrapped.observation_space
        self.action_space = env.unwrapped.action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.bins = bins
        self.inventory_intervals = np.linspace(0, env.unwrapped.max_demand*2, num=bins)
        self.demand_intervals = np.linspace(0, env.unwrapped.max_demand, num=bins)
        self.price_intervals = np.linspace(env.unwrapped.min_price, env.unwrapped.max_price, num=bins)
        self.quantity_intervals = np.linspace(0, env.unwrapped.max_quantity, num=bins)

        # Initialize Q-table as a dictionary
        self.q_table = {}

    def _get_state_key(self, state):
        inventory_level = tuple(np.digitize(state["inventory"], self.inventory_intervals))
        # demand_level = tuple(np.digitize(state["demand"], self.demand_intervals))
        # price_level = tuple([tuple(np.digitize(price, self.price_intervals)) for price in state["price"]])
        # quantity_level = tuple([tuple(np.digitize(quantity, self.quantity_intervals)) for quantity in state["quantity"]])
        return inventory_level
    
    def _get_action_key(self, action):
        return tuple((i, j, action[i, j]) for i in range(self.env.unwrapped.num_suppliers) for j in range(self.env.unwrapped.num_products))
    
    def _get_action_space(self):
        combinations = list(itertools.product(range(self.env.unwrapped.num_suppliers), range(self.env.unwrapped.num_products)))
        buy_combinations = list(itertools.product(range(self.env.unwrapped.buy_bins + 1), repeat=self.env.unwrapped.num_suppliers*self.env.unwrapped.num_products))
        space = []
        for b in buy_combinations:
            action_list = list(zip(combinations, b))
            action_items = ((i, j, k) for (i, j), k in action_list)
            space.append(tuple(action_items))
        return space
    
    def choose_action(self, state, greedy=False):
        state_key = self._get_state_key(state)
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_prob and not greedy:
            return self.action_space.sample()  # Explore
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = {action_key: 0 for action_key in self._get_action_space()}
            action_key = max(self.q_table[state_key], key=self.q_table[state_key].get)
            action = np.zeros((self.env.unwrapped.num_suppliers, self.env.unwrapped.num_products), dtype=int)
            # Populating the array with values from the tuple
            for i, j, k in action_key:
                action[i][j] = k
            return action

    def update(self, state, action, next_state, reward):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        action_key = self._get_action_key(action)

        if state_key not in self.q_table:
            self.q_table[state_key] = {action_key: 0 for action_key in self._get_action_space()}

        max_action_key = max(self.q_table[next_state_key], key=self.q_table[next_state_key].get)
        max_action = 0 if next_state_key not in self.q_table else self.q_table[next_state_key][max_action_key]
        # Q-value update using the Q-learning formula
        self.q_table[state_key][action_key] = (
            1 - self.learning_rate
        ) * self.q_table[state_key][action_key] + self.learning_rate * (
            reward + self.discount_factor * max_action
        )
        

    def train(self, num_episodes=1000, load=False, save=False, filename="q_table.npy"):
        if load:
            self.q_table = np.load(filename, allow_pickle=True).item()
        else:
            for _ in range(num_episodes):
                state, info = self.env.unwrapped.reset()
                while True:
                    action = self.choose_action(state)
                    next_state, reward, terminated, truncated, info = self.env.unwrapped.step(action)
                    self.update(state, action, next_state, reward)
                    state = next_state
                    if terminated or truncated:
                        break
            if save:
                np.save(filename, self.q_table)