from mip import MIP
import gymnasium
import supply_chain


env = gymnasium.make('supply_chain/SupplyChain-v0', action_mode='continuous')

demand, price, quantity, sell_price, transport_time = env.generate_random_data(seed=42)
mip = MIP(env, demand, sell_price, price, quantity, transport_time)
mip.run_model()