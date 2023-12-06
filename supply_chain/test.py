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
import copy

# Use just one agent to see training curve (add seed to the train)
# Use multiple agents to compare using box plots
agents = {
    Agents.QLEARNINGSS: {
        "name": "QLEARNINGSS",
        "run": False,
        "num_episodes": 100000,
    },
    Agents.QLEARNING: {
        "name": "QLEARNING",
        "run": False,
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
        "num_episodes": 5000,
    },
}
seed = 42
episodes = range(1)
comparison = True

agents_rewards = []
agents_names = []

for agent_name in agents.keys():
    single_rewards = [] 
    rewards = []

    if agent_name == Agents.QLEARNINGSS and agents[agent_name]["run"]:
        env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')
        agent = QLearningSSAgent(env)
        single_rewards_ss = agent.train(num_episodes=agents[agent_name]["num_episodes"], seed=seed, save=False, load=True, filename="q_table_ss_250000.npy", train=False)
        # print(agent.q_table)
        # print({outer_key: max(inner_dict, key=inner_dict.get) if inner_dict else None for outer_key, inner_dict in agent.q_table.items()})
    elif agent_name == Agents.QLEARNING and agents[agent_name]["run"]:
        env = gymnasium.make('supply_chain/SupplyChain-v0')
        agent = QLearningAgent(env)
        single_rewards = agent.train(num_episodes=agents[agent_name]["num_episodes"], seed=seed, save=False, load=True, filename="q_table_250000.npy", train=False)
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
        rewards_df["Moving Average"] = rewards_df["Reward"].rolling(window=1500).mean()
        plt.plot(rewards_df["Moving Average"], label="Q Learning", color="red")
        
        # plt.axhline(y=1463.68, xmin=0, xmax=len(single_rewards), linestyle="--", color='black')
        # plt.axhline(y=-569.17, xmin=0, xmax=len(single_rewards), linestyle="--", color='black')
        # plt.title("Convergencia de los agentes de Q Learning (Aleatoria)")
        # rewards_df = pd.DataFrame(single_rewards_ss, columns=["Reward"])
        # rewards_df["Moving Average"] = rewards_df["Reward"].rolling(window=1500).mean()
        # plt.plot(rewards_df["Moving Average"], label="Q Learning SS", color="b")
        # plt.grid(axis = 'y')
        # plt.locator_params(nbins=10)
        
        # plt.legend()
        # plt.xlabel("Episodios")
        # plt.ylabel("Recompensa ($)")

        # plt.savefig(f'QLEARNING vs QLEARNINGSS-Random-42-convergence.png')
        
        plt.savefig(f'assets/{agents[agent_name]["name"]}-{seed}-{agents[agent_name]["num_episodes"]}-{time.strftime("%H-%M-%S", time.localtime())}.png')
        # plt.show()

    if comparison and agents[agent_name]["run"]:
        state_array = []
        action_array = []
        rewards_array = []
        for _ in episodes:
            state, info = env.reset()
            score = 0
            while True:
                print(env.transport_time)
                action = agent.choose_action(state, greedy=True)
                state_array.append(copy.deepcopy(state))
                action_array.append(copy.deepcopy(action))
                next_state, reward, terminated, truncated, info = env.step(action)
                rewards_array.append(reward)
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

if len(agents_rewards) > 1 and comparison:
    fig = plt.figure(figsize=(12, 6))
    plt.boxplot(agents_rewards, showfliers=False, labels=agents_names)
    plt.grid(axis = 'y')
    plt.locator_params(nbins=10)
    
    plt.title("Comparación de los agentes (Aleatoria)")
    plt.ylabel("Recompensa ($)")

    plt.savefig(f'assets/{"_".join(agents_names)}-{len(episodes)}-{time.strftime("%H-%M-%S", time.localtime())}.png')
    # plt.show()
    
prices1 =  [obj['price'][0][0] for obj in state_array]
prices2 =  [obj['price'][1][0] for obj in state_array]
quantities1 =  [obj['quantity'][0][0] for obj in state_array]
quantities2 =  [obj['quantity'][1][0] for obj in state_array]
demand =  [obj['demand'][0] for obj in state_array]
inventory = [obj['inventory'][0] for obj in state_array]
buy1 = [obj[0][0]/7 for obj in action_array]
buy2 = [obj[1][0]/7 for obj in action_array]


fig = plt.figure(figsize=(12, 6))
data1_df = pd.DataFrame(prices1, columns=["data"])
data2_df = pd.DataFrame(prices2, columns=["data"])
plt.plot(data1_df["data"], label="Proveedor 1", color="red")
plt.plot(data2_df["data"], label="Proveedor 2", color="b")
plt.grid(axis = 'y')
plt.locator_params(nbins=10)
plt.legend()
plt.xlabel("Periodos")
plt.ylabel("Precio por cada 100kg ($)")
plt.title("Precios de los proveedores")
plt.savefig(f'Precios42.png')

fig = plt.figure(figsize=(12, 6))
data1_df = pd.DataFrame(quantities1, columns=["data"])
data2_df = pd.DataFrame(quantities2, columns=["data"])
plt.plot(data1_df["data"], label="Proveedor 1", color="red")
plt.plot(data2_df["data"], label="Proveedor 2", color="b")
plt.grid(axis = 'y')
plt.locator_params(nbins=10)
plt.legend()
plt.xlabel("Periodos")
plt.ylabel("Cantidades disponibles(kg)")
plt.title("Cantidades de los proveedores")
plt.savefig(f'Cantidades42.png')

fig = plt.figure(figsize=(12, 6))
data1_df = pd.DataFrame(demand, columns=["data"])
plt.plot(data1_df["data"], color="red")
plt.grid(axis = 'y')
plt.locator_params(nbins=10)
plt.xlabel("Periodos")
plt.ylabel("Demanda (kg)")
plt.title("Demanda del producto")
plt.savefig(f'Demandas42.png')

fig = plt.figure(figsize=(12, 6))
data1_df = pd.DataFrame(inventory, columns=["data"])
plt.plot(data1_df["data"], color="red")
plt.grid(axis = 'y')
plt.locator_params(nbins=10)
plt.xlabel("Periodos")
plt.ylabel("Inventario (kg)")
plt.title("Inventario de los proveedores")
plt.savefig(f'Inventarios42.png')

fig = plt.figure(figsize=(12, 6))
data1_df = pd.DataFrame(rewards_array, columns=["data"])
plt.plot(data1_df["data"], color="red")
plt.grid(axis = 'y')
plt.locator_params(nbins=10)
plt.xlabel("Periodos")
plt.ylabel("Recompensas ($)")
plt.title("Recompensas del agente")
plt.savefig(f'Recompensas42.png')

fig = plt.figure(figsize=(12, 6))
data1_df = pd.DataFrame(buy1, columns=["data"])
data2_df = pd.DataFrame(buy2, columns=["data"])
plt.plot(data1_df["data"], label="Proveedor 1", color="red")
plt.plot(data2_df["data"], label="Proveedor 2", color="b")
plt.grid(axis = 'y')
plt.locator_params(nbins=10)
plt.legend()
plt.xlabel("Periodos")
plt.ylabel("Porcentaje comprado (%)")
plt.title("Compras a los proveedores")
plt.savefig(f'Compras42.png')