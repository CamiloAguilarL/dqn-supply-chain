import gymnasium
import numpy as np
from agent_interface import AgentInterface

class RQAgent(AgentInterface):
    def __init__(self, env, reorder_percentage=0.5, order_percentage=2):
        self.env = env
        self.reorder_point = reorder_percentage*env.max_demand
        self.order_quantity = order_percentage*env.max_demand

    def choose_action(self, state, greedy=True):
        inventory = state['inventory']  # Assuming observation is the inventory level
        price = state['price']
        quantity = state['quantity']
        action = np.zeros((self.env.num_suppliers, self.env.num_products), dtype=int)
        for i, product in enumerate(inventory):
            if product < self.reorder_point:
                visited_suppliers = []
                total_quantity = 0
                while len(visited_suppliers) < self.env.num_suppliers:
                    min_price = self.env.max_price * 10
                    min_price_index = -1
                    for j, supplier in enumerate(price):
                        if supplier[i] < min_price and j not in visited_suppliers:
                            min_price = supplier[i]
                            min_price_index = j
                    visited_suppliers.append(min_price_index)
                    buy_quantity = min(self.order_quantity - total_quantity, quantity[min_price_index, i])
                    total_quantity += buy_quantity
                    action[min_price_index,i] = buy_quantity
                    if total_quantity >= self.order_quantity:
                        break

        return action
    
    def update(self, state, action, next_state, reward):
        pass
        

    def train(self, num_episodes):
        pass
    