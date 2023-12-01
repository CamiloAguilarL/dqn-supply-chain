import numpy as np
import itertools
import copy

from agent_interface import AgentInterface

class QLearningAgent(AgentInterface):
    def __init__(self, env, bins=5, learning_rate=0.2, discount_factor=0.8, exploration_prob=0.3, min_exploration_prob=0.01, decay_rate=0.00002):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.decay_rate = decay_rate
        self.bins = bins
        self.inventory_intervals = np.linspace(0, env.max_demand, num=bins)
        self.demand_intervals = np.linspace(env.min_demand, env.max_demand, num=bins)
        self.price_intervals = np.linspace(env.min_price, env.max_price, num=bins) 
        self.quantity_intervals = np.linspace(env.min_quantity, env.max_quantity, num=bins)
        self.transport_intervals = np.linspace(env.min_transport_time, env.max_transport_time, num=bins)
        # Initialize Q-table as a dictionary
        self.q_table = {}

    def _get_state_key(self, state):
        inventory_level = tuple(np.digitize(state["inventory"], self.inventory_intervals))
        demand_level = tuple(np.digitize(state["demand"], self.demand_intervals))
        transport_level = tuple(np.digitize(self.env.transport_time, self.transport_intervals))
        price_level = tuple([tuple(np.digitize(price, self.price_intervals)) for price in state["price"]])
        quantity_level = tuple([tuple(np.digitize(quantity, self.quantity_intervals)) for quantity in state["quantity"]])
        return inventory_level, demand_level, transport_level, price_level, quantity_level
    
    def _get_action_key(self, action):
        return tuple((i, j, action[i, j]) for i in range(self.env.num_suppliers) for j in range(self.env.num_products))
    
    def _get_action_space(self):
        combinations = list(itertools.product(range(self.env.num_suppliers), range(self.env.num_products)))
        buy_combinations = list(itertools.product(range(self.env.buy_bins + 1), repeat=self.env.num_suppliers*self.env.num_products))
        space = []
        for b in buy_combinations:
            action_list = list(zip(combinations, b))
            action_items = ((i, j, k) for (i, j), k in action_list)
            space.append(tuple(action_items))
        return space
    
    def choose_action(self, state, greedy=False):
        state_key = self._get_state_key(state)
        # print(state_key)
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_prob and not greedy:
            if state_key not in self.q_table:
                self.q_table[state_key] = {action_key: 0 for action_key in self._get_action_space()}
            random_index = np.random.randint(0, len(self.q_table[state_key].keys()))
            action_key = list(self.q_table[state_key].keys())[random_index]
            action = np.zeros((self.env.num_suppliers, self.env.num_products), dtype=int)
            for i, j, k in action_key:
                action[i][j] = k
            return action
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = {action_key: 0 for action_key in self._get_action_space()}
            action_key = max(self.q_table[state_key], key=self.q_table[state_key].get)
            action = np.zeros((self.env.num_suppliers, self.env.num_products), dtype=int)
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
        rewards = []
        if load:
            self.q_table = np.load(filename, allow_pickle=True).item()
        if train:
            for _ in range(num_episodes):
                state, info = self.env.reset(seed=seed)
                score = 0
                while True:
                    action = self.choose_action(state)
                    prev_state = copy.deepcopy(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    self.update(prev_state, action, next_state, reward)
                    state = next_state
                    score += info["objective"]
                    if terminated or truncated:
                        break
                rewards.append(score)
                self.exploration_prob = max(self.exploration_prob * (1-self.decay_rate), self.min_exploration_prob)
                if _ % 100 == 0:
                    print(f"Training Agent: Q Learning, Episode: {_}, Score: {score}")
            if save:
                np.save(filename, self.q_table)

        return rewards

    
    def test_seed(self, load=True, filename="q_table.npy", seed=None):

        if load:
            self.q_table = np.load(filename, allow_pickle=True).item()

        state, info = self.env.reset(seed=seed)
        score = 0
        while True:
            action = self.choose_action(state, greedy=True)
            prev_state = copy.deepcopy(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # print(action)
            # print(state)
            # print("==========================")
            state = next_state
            score += info["objective"]
            if terminated or truncated:
                break
        
        print(score)