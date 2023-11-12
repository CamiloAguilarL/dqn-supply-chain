from gymnasium.envs.registration import register

register(
    id="grid_world/GridWorld-v0",
    entry_point="grid_world.envs:GridWorldEnv",
)
