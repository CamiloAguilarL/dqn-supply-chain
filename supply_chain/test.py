import gymnasium
import supply_chain
from qlearning import QLearningAgent
import matplotlib.pyplot as plt

demands = []
inventory = []
rewards = []

env = gymnasium.make('supply_chain/SupplyChain-v0')
agent = QLearningAgent(env)
agent.train(10000)
agent.set_exploration_prob(0)
for _ in range(1):
    demands = []
    rewards = []
    inventory = []
    state, info = env.reset()
    while True:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update_q_table(state, action, next_state, reward)
        demands.append(state['demand'][0])
        inventory.append(state['inventory'][0])
        rewards.append(reward)

        state = next_state
        if terminated or truncated:
            break
env.close()

episodes = range(1, 10 + 1)
plt.figure(figsize=(12, 6))
plt.plot(episodes, demands, label='Average Demand')
plt.plot(episodes, inventory, label='Average Inventory')
# plt.plot(episodes, rewards, label='Average Rewards')
plt.xlabel('Episodes')
plt.ylabel('Amount')
plt.title('Demand, Inventory, and Purchase for Product over Episodes')
plt.legend()
plt.show()