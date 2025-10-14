from functools import partial

from .multiagentenv import MultiAgentEnv
from .particle import Particle
# from .mamujoco import ManyAgentAntEnv, ManyAgentSwimmerEnv, MujocoMulti

def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)

REGISTRY = {}
REGISTRY["particle"] = partial(env_fn, env=Particle)
# REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
# REGISTRY["manyagent_swimmer"] = partial(env_fn, env=ManyAgentSwimmerEnv)
# REGISTRY["manyagent_ant"] = partial(env_fn, env=ManyAgentAntEnv)
