from agent_interface import AgentInterface
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from collections import deque
import itertools
import numpy as np
import random
import copy


class DQNAgent(AgentInterface):
    def __init__(self, env, learning_rate=0.2, discount_factor=0.9, exploration_prob=0.3, min_exploration_prob=0.01, decay_rate=0.00005):
        self.env = env
        learning_rate = 0.001
        sample_observation = env.observation_space.sample()
        flattened_observation = np.concatenate([sample_observation[key].flatten() for key in sample_observation])
        input_shape = flattened_observation.shape
        actions = (env.buy_bins + 1) ** (env.action_space.shape[0] * env.action_space.shape[1])

        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        model.compile(loss=Huber(), optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        self.model = model

        target_model = Sequential()
        target_model.add(Flatten(input_shape=input_shape))
        target_model.add(Dense(16, activation='relu'))
        target_model.add(Dense(16, activation='relu'))
        target_model.add(Dense(actions, activation='linear'))
        target_model.compile(loss=Huber(), optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        target_model.set_weights(model.get_weights())
        self.target_model = target_model

        self.replay_memory = deque(maxlen=50_000)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.decay_rate = decay_rate

        combinations = list(itertools.product(range(self.env.num_suppliers), range(self.env.num_products)))
        buy_combinations = list(itertools.product(range(self.env.buy_bins + 1), repeat=self.env.num_suppliers*self.env.num_products))
        space = []
        for b in buy_combinations:
            action_list = list(zip(combinations, b))
            action_items = ((i, j, k) for (i, j), k in action_list)
            space.append(tuple(action_items))

        self.actions_array = space

    def update(self, state, action, next_state, reward):

        MIN_REPLAY_SIZE = 10
        if len(self.replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 8
        mini_batch = random.sample(self.replay_memory, batch_size)
        current_states = np.array([self._get_state_array(transition[0]) for transition in mini_batch])
        current_qs_list = self.model.predict(current_states, verbose = 0)

        new_current_states = np.array([self._get_state_array(transition[3]) for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states, verbose = 0)

        X = []
        Y = []

        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            max_future_q = reward + self.discount_factor * np.max(future_qs_list[index])
            action_index = self._get_action_index(action)
            current_qs = current_qs_list[index]
            current_qs[action_index] = (1 - self.learning_rate) * current_qs[action_index] + self.learning_rate * max_future_q
            observation = self._get_state_array(observation)
            X.append(observation)
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

    def _get_state_array(self, state):
        return np.concatenate([state[key].flatten() for key in state])

    def _get_action_index(self, action):
        action_key = tuple((i, j, action[i, j]) for i in range(self.env.num_suppliers) for j in range(self.env.num_products))
        return self.actions_array.index(action_key)
    
    def choose_action(self, state, greedy=False):
        state_array = self._get_state_array(state).reshape(1, -1)
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_prob and not greedy:
            action_index = np.random.randint(0, len(self.actions_array))
            action_key = self.actions_array[action_index]
        else:
            action_index = self.model.predict(state_array, verbose = 0).flatten()
            action = np.argmax(action_index)
            action_key = self.actions_array[action]

        action = np.zeros((self.env.num_suppliers, self.env.num_products), dtype=int)
        for i, j, k in action_key:
            action[i][j] = k
        return action


    def train(self, num_episodes=1000, load=False, save=False, train=True, filename="dqn.keras", seed=None):
        rewards = []
        if load:
            self.model = load_model(filename)
            self.target_model.set_weights(self.model.get_weights())
        if train:
            update_target = 0

            for episode in range(num_episodes):
                state, info = self.env.reset(seed=seed)
                score = 0
                while True:
                    update_target += 1
                    action = self.choose_action(state)
                    prev_state = copy.deepcopy(state)                
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    self.replay_memory.append([state, action, reward, next_state, terminated])
                    if update_target % 2 == 0 or terminated:
                        self.update(prev_state, action, next_state, reward)
                    state = next_state
                    score += reward
                    if terminated or truncated:
                        if update_target >= 25:
                            self.target_model.set_weights(self.model.get_weights())
                            update_target = 0
                        break
                rewards.append(score)
                self.exploration_prob = max(self.exploration_prob * (1-self.decay_rate), self.min_exploration_prob)
                if episode % 10 == 0:
                    print(f"Training Agent: DQN, Episode: {episode}, Score: {score}")
            if save:
                self.model.save(filename)

        return rewards
