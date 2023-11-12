import gymnasium
import grid_world
from qlearning import QLearningAgent
import numpy as np

train_env = gymnasium.make('grid_world/GridWorld-v0', render_mode=None, size=3)
env = gymnasium.make('grid_world/GridWorld-v0', render_mode="human", size=3, render_fps=3)

train_env.action_space.seed(42)
env.action_space.seed(42)

agent = QLearningAgent(train_env)
agent.train(1000)
agent.set_exploration_prob(0)
state, info = env.reset(seed=42)

for _ in range(10):
    state, info = env.reset()
    while True:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update_q_table(state, action, next_state, reward)
        state = next_state
        if terminated or truncated:
            break

env.close()