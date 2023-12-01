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
        "run": True,
        "num_episodes": 250000,
    },
    Agents.QLEARNING: {
        "name": "QLEARNING",
        "run": True,
        "num_episodes": 250000,
    },
    Agents.RQ: {
        "name": "RQ",
        "run": True,
    },
    Agents.SS: {
        "name": "SS",
        "run": True,
    },
    Agents.DQN: {
        "name": "DQN",
        "run": True,
        "num_episodes": 5000,
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
        single_rewards = agent.train(num_episodes=agents[agent_name]["num_episodes"], seed=seed, save=False, load=True, filename="q_table_ss.npy", train=False)
        # print(agent.q_table)
        # print({outer_key: max(inner_dict, key=inner_dict.get) if inner_dict else None for outer_key, inner_dict in agent.q_table.items()})
    elif agent_name == Agents.QLEARNING and agents[agent_name]["run"]:
        env = gymnasium.make('supply_chain/SupplyChain-v0')
        agent = QLearningAgent(env)
        single_rewards = agent.train(num_episodes=agents[agent_name]["num_episodes"], seed=seed, save=False, load=True, filename="q_table.npy", train=False)
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
        single_rewards = agent.train(num_episodes=agents[agent_name]["num_episodes"], seed=seed, save=False, load=True, train=False)

    if len(single_rewards) > 0:
        fig = plt.figure(figsize=(12, 6))
        rewards_df = pd.DataFrame(single_rewards, columns=["Reward"])
        rewards_df["Moving Average"] = rewards_df["Reward"].rolling(window=100).mean()
        plt.plot(rewards_df["Moving Average"], color="red")
        
        # plt.axhline(y=1463.68, xmin=0, xmax=len(single_rewards), linestyle="--", color='black')
        # plt.axhline(y=275.41763821158077, xmin=0, xmax=len(single_rewards), linestyle="--", color='black')
        # plt.title("Convergencia de los agentes de Q Learning (Aleatoria)")
        # rewards_df = pd.DataFrame(single_rewards_ss, columns=["Reward"])
        # rewards_df["Moving Average"] = rewards_df["Reward"].rolling(window=1500).mean()
        # plt.plot(rewards_df["Moving Average"], label="Q Learning SS", color="b")
        # plt.grid(axis = 'y')
        # plt.locator_params(nbins=10)
        
        # plt.legend()
        # plt.xlabel("Episodios")
        # plt.ylabel("Recompensa ($)")

        # plt.savefig(f'QLEARNING vs QLEARNINGSS-Random.png')
        
        plt.savefig(f'assets/{agents[agent_name]["name"]}-{seed}-{agents[agent_name]["num_episodes"]}-{time.strftime("%H-%M-%S", time.localtime())}.png')
        # plt.show()

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
    plt.boxplot(agents_rewards, showfliers=False, labels=agents_names)
    plt.grid(axis = 'y')
    plt.locator_params(nbins=10)
    
    plt.title("Comparación de los agentes (Aleatoria)")
    plt.ylabel("Recompensa ($)")

    plt.savefig(f'assets/{"_".join(agents_names)}-{len(episodes)}-{time.strftime("%H-%M-%S", time.localtime())}.png')
    # plt.show()