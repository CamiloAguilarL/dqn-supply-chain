from agent_interface import AgentInterface
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from collections import deque
import random

import numpy as np

class DQNAgent(AgentInterface):
    def __init__(self, env):
        self.env = env
        learning_rate = 0.001
        sample_observation = env.observation_space.sample()
        flattened_observation = np.concatenate([sample_observation[key].flatten() for key in sample_observation])
        input_shape = flattened_observation.shape
        actions = env.buy_bins ** (env.action_space.shape[0] * env.action_space.shape[1])

        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        model.compile(loss=Huber(), optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        self.model = model

        target_model = Sequential()
        target_model.add(Flatten(input_shape=input_shape))
        target_model.add(Dense(64, activation='relu'))
        target_model.add(Dense(64, activation='relu'))
        target_model.add(Dense(actions, activation='linear'))
        target_model.compile(loss=Huber(), optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        target_model.set_weights(model.get_weights())
        self.target_model = target_model

    def get_qs(self, state):
        return self.model.predict(state.reshape([1, state.shape[0]]))[0]

    def train(self, replay_memory, done):
        learning_rate = 0.7 # Learning rate
        discount_factor = 0.618

        MIN_REPLAY_SIZE = 1000
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64 * 2
        mini_batch = random.sample(replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(observation)
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


    def choose_action(self, state, greedy=True):
        pass
    
    def update(self, state, action, next_state, reward):
        pass
        

    def train(self, num_episodes=1000, load=False, save=False, train=True, filename="q_table.npy", seed=None):
        epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
        max_epsilon = 1 # You can't explore more than 100% of the time
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
        decay = 0.01

        replay_memory = deque(maxlen=50_000)

        # X = states, y = actions
        X = []
        y = []

        steps_to_update_target_model = 0

        for episode in range(num_episodes):
            total_training_rewards = 0
            state, info = self.env.reset()
            done = False
            while True:
                steps_to_update_target_model += 1

                random_number = np.random.rand()
                # 2. Explore using the Epsilon Greedy Exploration Strategy
                if random_number <= epsilon:
                    # Explore
                    action = self.env.action_space.sample()
                else:
                    encoded = state
                    encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                    predicted = self.model.predict(encoded_reshaped).flatten()
                    action = np.argmax(predicted)
                
                next_state, reward, done, info = self.env.step(action)
                replay_memory.append([state, action, reward, next_state, done])

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0 or done:
                    self.train(replay_memory, done)

                state = next_state
                total_training_rewards += reward

                if done:
                    print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                    total_training_rewards += 1

                    if steps_to_update_target_model >= 100:
                        print('Copying main network weights to the target network weights')
                        self.target_model.set_weights(self.model.get_weights())
                        steps_to_update_target_model = 0
                    break

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        env.close()
