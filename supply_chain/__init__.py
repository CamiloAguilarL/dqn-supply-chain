from gymnasium.envs.registration import register

register(
    id="supply_chain/SupplyChain-v0",
    entry_point="supply_chain.envs:SupplyChainEnv",
)
