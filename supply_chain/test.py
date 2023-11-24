import gymnasium
import supply_chain

from agents import Agents
from rq_greedy import RQAgent
from ss_greedy import SSAgent
from q_learning_ss import QLearningSSAgent
from q_learning import QLearningAgent

import matplotlib.pyplot as plt
import pandas as pd
import time

# Use just one agent to see training curve (add seed to the train)
# Use multiple agents to compare using box plots
agents = {
    Agents.QLEARNINGSS: {
        "name": "QLEARNINGSS",
        "run": False,
        "num_episodes": 50000,
    },
    Agents.QLEARNING: {
        "name": "QLEARNING",
        "run": True,
        "num_episodes": 100000,
    },
    Agents.RQ: {
        "name": "RQ",
        "run": False,
    },
    Agents.SS: {
        "name": "SS",
        "run": False,
    },
    Agents.DQN: {
        "name": "DQN",
        "run": True,
        "num_episodes": 10000,
    },
}
seed = None
episodes = range(100)
comparison = True

agents_rewards = []
agents_names = []

for agent_name in agents.keys():
    single_rewards = []
    rewards = []

    if agent_name == Agents.QLEARNINGSS and agents[agent_name]["run"]:
        env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')
        agent = QLearningSSAgent(env)
        single_rewards = agent.train(num_episodes=agents[agent_name]["num_episodes"], seed=seed)
        # print(agent.q_table)
        # print({outer_key: max(inner_dict, key=inner_dict.get) if inner_dict else None for outer_key, inner_dict in agent.q_table.items()})
    elif agent_name == Agents.QLEARNING and agents[agent_name]["run"]:
        env = gymnasium.make('supply_chain/SupplyChain-v0')
        agent = QLearningAgent(env)
        single_rewards = agent.train(num_episodes=agents[agent_name]["num_episodes"], seed=seed, save=False, load=True, filename="q_table_100000.npy", train=False)
        # agent.test_seed(seed=seed, filename="q_table_50000.npy")
        # print(agent.q_table)
        # print({outer_key: max(inner_dict, key=inner_dict.get) if inner_dict else None for outer_key, inner_dict in agent.q_table.items()})
    elif agent_name == Agents.RQ and agents[agent_name]["run"]:
        env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')
        agent = RQAgent(env)
    elif agent_name == Agents.SS and agents[agent_name]["run"]:
        env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')
        agent = SSAgent(env)
    elif agent_name == Agents.DQN and agents[agent_name]["run"]:
        from dqn import DQNAgent
        env = gymnasium.make('supply_chain/SupplyChain-v0')
        agent = DQNAgent(env)
        single_rewards = agent.train(num_episodes=agents[agent_name]["num_episodes"], seed=seed, save=True, load=True, train=True)

    if len(single_rewards) > 0:
        fig = plt.figure(figsize=(12, 6))
        rewards_df = pd.DataFrame(single_rewards, columns=["Reward"])
        rewards_df["Moving Average"] = rewards_df["Reward"].rolling(window=5).mean()
        plt.plot(rewards_df["Moving Average"], label="Moving Average (10)", color="red")
        plt.savefig(f'{agents[agent_name]["name"]}-{seed}-{agents[agent_name]["num_episodes"]}-{time.strftime("%H-%M-%S", time.localtime())}.png')
        plt.show()

    if comparison and agents[agent_name]["run"]:
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
            rewards.append(score)
            if _ % 100 == 0:
                print(f"Agent: {agent_name}, Episode: {_}, Score: {score}")
        agents_rewards.append(rewards)
        agents_names.append(agents[agent_name]["name"])
    
    if agents[agent_name]["run"]:
        env.close()

if len(agents_rewards) > 1:
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.boxplot(agents_rewards, showfliers=False)
    plt.savefig(f'{"_".join(agents_names)}-{len(episodes)}-{time.strftime("%H-%M-%S", time.localtime())}.png')
    plt.show()