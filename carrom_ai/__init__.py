from gymnasium.envs.registration import register

register(
     id="carrom_ai/CarromGym-v0",
     entry_point="carrom_ai.envs:carrom_env",
)