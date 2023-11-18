import gymnasium
import supply_chain

from agents import Agents
from rq_greedy import RQAgent
from ss_greedy import SSAgent
from q_learning import QLearningAgent

import matplotlib.pyplot as plt

agents = [Agents.QLEARNING, Agents.RQ, Agents.SS]

seed = 42
episodes = range(1000)

agents_rewards = []
for agent_name in agents:
    rewards = []

    if agent_name == Agents.QLEARNING:
        env = gymnasium.make('supply_chain/SupplyChain-v0')
        agent = QLearningAgent(env)
        # agent.train(num_episodes=5000)
    elif agent_name == Agents.RQ:
        env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')
        agent = RQAgent(env)
    elif agent_name == Agents.SS:
        env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')
        agent = SSAgent(env)

    for _ in episodes:
        state, info = env.reset()
        score = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, next_state, reward)
            score += reward

            state = next_state
            if terminated or truncated:
                break
        rewards.append(score)
        if _ % 100 == 0:
            print(f"Agent: {agent_name}, Episode: {_}, Score: {score}")
    agents_rewards.append(rewards)
env.close()

fig = plt.figure(figsize=(12, 6))
# plt.plot(episodes, rewards, label='Rewards')

ax = fig.add_axes([0, 0, 1, 1])
ax.boxplot(agents_rewards)

plt.show()