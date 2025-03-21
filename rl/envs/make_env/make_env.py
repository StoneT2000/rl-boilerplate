"""
All in one file to make most environments from e.g. D4RL to maniskill to whatever
"""

from typing import Callable
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from gymnasium.vector import VectorEnv
import numpy as np
import torch
from rl.envs import gym_utils
from rl.envs.make_env import mani_skill3
@dataclass
class EnvMeta:
    num_envs: int
    sample_obs: torch.Tensor
    sample_acts: torch.Tensor
    obs_space: spaces.Space
    act_space: spaces.Space
    env_suite: str
    max_episode_steps: int | None = None
    reward_mode: str | None = None
    """a bit of an arbitrary tag. Just general concept of whether it is dense, sparse, or something else for tracking purposes"""

@dataclass
class EnvConfig:
    env_id: str
    """environment id, passed to gym.make"""
    vectorization_method: str
    """Can be "cpu" or "gpu" or "jax". CPU will try and use CPU vectorization. GPU will try and use GPU vectorization. Jax will use JAX vectorization."""
    max_episode_steps: int | None = None
    """max episode steps for the environment. If none will use what is defined in the environment"""
    num_envs: int = 1
    """number of parallel environments to create"""
    seed: int | list[int] | None = None
    """seed for the environment"""
    ignore_terminations: bool = False
    """if true, will ignore terminations and continue the episode until truncation. If false, will stop at termination and auto reset"""
    auto_reset: bool = True
    """if true, will auto reset the environment when the episode is done"""

    env_kwargs: dict = field(default_factory=dict)
    """additional kwargs to pass to the environment constructor"""

    record_video_path: str | None = None
    """the path to record videos to. If None, no videos will be recorded."""
    record_episode_kwargs: dict = field(default_factory=dict)
    """the kwargs to record the episode with"""

def make_env_from_config(env_config: EnvConfig, wrappers: list[Callable[[gym.Env], gym.Wrapper]] = []) -> tuple[VectorEnv, EnvMeta]:
    return make_env(
        env_config.env_id,
        vectorization_method=env_config.vectorization_method,
        max_episode_steps=env_config.max_episode_steps,
        num_envs=env_config.num_envs,
        seed=env_config.seed,
        ignore_terminations=env_config.ignore_terminations,
        auto_reset=env_config.auto_reset,
        env_kwargs=env_config.env_kwargs,
        record_video_path=env_config.record_video_path,
        record_episode_kwargs=env_config.record_episode_kwargs,
        wrappers=wrappers,
    )
def make_env(
    env_id: str,
    vectorization_method: str = "cpu",
    max_episode_steps: int | None = None,
    num_envs: int = 1,
    seed: int | list[int] | None = 0,
    ignore_terminations: bool = False,
    auto_reset: bool = True,
    env_kwargs: dict = dict(),
    record_video_path: str | None = None,
    record_episode_kwargs: dict = dict(),
    wrappers=[],
) -> tuple[VectorEnv, EnvMeta]:
    """
    Make a gymnasium vector env of the given environment id.

    Args:
        env_id: the id of the environment to make
        vectorziation_method: the method to vectorize the environment, either "cpu" or "gpu" or "jax". CPU will try and use CPU vectorization. GPU will try and use GPU vectorization. Jax will use JAX vectorization.
            Not all environments support all vectorization methods (most only support 1 or 2).
        max_episode_steps: the maximum number of steps in each episode. If none will use what is defined in the environment.
        num_envs: the number of parallel environments to make
        seed: the seed for the environment on the first reset
        env_kwargs: the kwargs to pass to the environment
        record_video_path: the path to record videos to. If None, no videos will be recorded.
        record_episode_kwargs: the kwargs to record the episode with
        wrappers: the wrappers to apply to the environment
    """
    env_suite = ""
    get_env_reward_mode_fn = None
    for pkg in [mani_skill3]:
        if pkg.has_env(env_id):
            if not pkg.supports_vectorization(vectorization_method):
                raise ValueError(f"Vectorization method {vectorization_method} not supported for {env_suite} environments")
            env_suite = pkg.SUITE_NAME
            if hasattr(pkg, "env_factory_cpu"):
                env_factory_cpu = pkg.env_factory_cpu
            if hasattr(pkg, "env_factory_gpu"):
                env_factory_gpu = pkg.env_factory_gpu
            if hasattr(pkg, "get_env_reward_mode"):
                get_env_reward_mode_fn = pkg.get_env_reward_mode
            break
    if env_suite == "":
        raise ValueError(f"Environment {env_id} not found")
    
    if vectorization_method == "cpu":
        vector_env_cls = gym.vector.AsyncVectorEnv if num_envs > 1 else gym.vector.SyncVectorEnv
        env: VectorEnv = vector_env_cls(
            [
                env_factory_cpu(
                    env_id,
                    idx,
                    env_kwargs=env_kwargs,
                    record_video_path=record_video_path,
                    wrappers=wrappers,
                    record_episode_kwargs=record_episode_kwargs,
                )
                for idx in range(num_envs)
            ]
        )
    elif vectorization_method == "gpu":
        env: VectorEnv = env_factory_gpu(
            env_id, 
            num_envs=num_envs, 
            seed=seed, 
            ignore_terminations=ignore_terminations, 
            auto_reset=auto_reset, 
            env_kwargs=env_kwargs, 
            record_video_path=record_video_path, 
            wrappers=wrappers, 
            record_episode_kwargs=record_episode_kwargs
        )


    obs_space = env.observation_space
    act_space = env.action_space
    # note some envs do not randomize assets, just poses if do this kind of reset
    obs, reset_info = env.reset(seed=seed)
    import tensordict
    sample_obs = tensordict.from_dict(obs)[0:1]
    sample_acts = act_space.sample()[0:1]

    max_episode_steps = gym_utils.find_max_episode_steps_value(env)

    return env, EnvMeta(
        num_envs=num_envs,
        obs_space=obs_space,
        act_space=act_space,
        sample_obs=sample_obs,
        sample_acts=sample_acts,
        env_suite=env_suite,
        max_episode_steps=max_episode_steps,
        reward_mode=get_env_reward_mode_fn(env) if get_env_reward_mode_fn is not None else None,
    )