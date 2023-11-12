import numpy as np

class QLearningAgent:
    def __init__(self, env, learning_rate=0.2, discount_factor=0.9, exploration_prob=0.1):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Initialize Q-table as a dictionary
        self.q_table = {}

    def _get_state_key(self, state):
        agent_key = tuple(state["agent"])
        target_key = tuple(state["target"])
        return agent_key, target_key

    def choose_action(self, state):
        state_key = self._get_state_key(state)

        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_prob:
            return self.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table.get(state_key, np.zeros(self.action_space.n)))  # Exploit

    def update_q_table(self, state, action, next_state, reward):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # Initialize Q-values for the current state if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)

        # Initialize Q-values for the next state if not present
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)

        # Q-value update using the Q-learning formula
        self.q_table[state_key][action] = (
            1 - self.learning_rate
        ) * self.q_table[state_key][action] + self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[next_state_key])
        )

    def train(self, num_episodes=1000):
        for _ in range(num_episodes):
            state, info = self.env.reset()
            while True:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.update_q_table(state, action, next_state, reward)
                state = next_state
                if terminated or truncated:
                    break
                
    def set_exploration_prob(self, prob):
        self.exploration_prob = prob