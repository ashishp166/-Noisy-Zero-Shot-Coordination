from environments import (
    SelfPlayNRLG,
    OtherPlayNZSC,
    GridEnv,
    MirrorEnv,
    BattleShipCommEnv,
    GridEnvComplex
)



def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")
    elif env_id == "SelfPlayNoisyLever":
        env = SelfPlayNRLG(**env_kwargs)
    elif env_id == "OtherPlay":
        env = OtherPlayNZSC(**env_kwargs)
    elif env_id == "Grid":
        env = GridEnv(**env_kwargs)
    elif env_id == "Mirror":
        env = MirrorEnv(**env_kwargs)
    elif env_id == "BattleShip":
        env = BattleShipCommEnv(**env_kwargs)
    elif env_id == "GridEnvComplex":
        env = GridEnvComplex(**env_kwargs)
    return env

registered_envs = [
    "simpleLever",
    "otherPlayNoisyLever",
    "SelfPlayNoisyLever",
    "OtherPlay",
    "Grid",
    "MiniGrid",
    "overcooked",
    "GridLarge",
    "Mirror",
    "CountEnv",
    "BattleShip",
    "GridEnvComplex"
]
