import gymnasium
import supply_chain

from agents import Agents
from rq_greedy import RQAgent
from ss_greedy import SSAgent
from q_learning_ss import QLearningSSAgent
from q_learning import QLearningAgent

import matplotlib.pyplot as plt
import pandas as pd

# Use just one agent to see training curve (add seed to the train)
# Use multiple agents to compare using box plots
agents = [Agents.QLEARNING, Agents.QLEARNINGSS]

seed = 42
episodes = range(100)

agents_rewards = []
single_rewards = []

for agent_name in agents:
    rewards = []

    if agent_name == Agents.QLEARNINGSS:
        env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')
        agent = QLearningSSAgent(env)
        single_rewards = agent.train(num_episodes=10, seed=seed)
        # print(agent.q_table)
        # print({outer_key: max(inner_dict, key=inner_dict.get) if inner_dict else None for outer_key, inner_dict in agent.q_table.items()})
    elif agent_name == Agents.QLEARNING:
        env = gymnasium.make('supply_chain/SupplyChain-v0')
        agent = QLearningAgent(env)
        single_rewards = agent.train(num_episodes=10, seed=seed)
        # print(agent.q_table)
        # print({outer_key: max(inner_dict, key=inner_dict.get) if inner_dict else None for outer_key, inner_dict in agent.q_table.items()})
    elif agent_name == Agents.RQ:
        env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')
        agent = RQAgent(env)
    elif agent_name == Agents.SS:
        env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')
        agent = SSAgent(env)
    elif agent_name == Agents.DQN:
        from dqn import DQNAgent
        env = gymnasium.make('supply_chain/SupplyChain-v0')
        agent = DQNAgent(env)
        single_rewards = agent.train(num_episodes=5000, seed=seed)

    if len(agents) > 1:
        for _ in episodes:
            state, info = env.reset()
            score = 0
            while True:
                action = agent.choose_action(state, greedy=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                score += reward
                state = next_state
                if terminated or truncated:
                    break
            rewards.append(score/env.num_periods)
            if _ % 50 == 0:
                print(f"Agent: {agent_name}, Episode: {_}, Score: {score}")
        agents_rewards.append(rewards)

    env.close()

fig = plt.figure(figsize=(12, 6))

if len(agents) <= 1:
    rewards_df = pd.DataFrame(single_rewards, columns=["Reward"])
    rewards_df["Moving Average"] = rewards_df["Reward"].rolling(window=100).mean()
    plt.plot(rewards_df["Moving Average"], label="Moving Average (10)", color="red")
else:
    ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(agents_rewards, showfliers=False)

plt.savefig('DQN-42-5000.png')
plt.show()