import gymnasium as gym
SUITE_NAME = "ManiSkill3"


try:
    import mani_skill.envs  # NOQA
    from mani_skill.utils.wrappers import RecordEpisode as RecordEpisodeWrapper
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
except ImportError:
    pass


def has_env(env_id: str):
    try:
        import mani_skill.envs  # NOQA
    except ImportError:
        return False
    from mani_skill.utils.registration import REGISTERED_ENVS

    return env_id in REGISTERED_ENVS
def supports_vectorization(vectorization_type: str):
    return vectorization_type in ["cpu", "gpu"]

def env_factory_cpu(env_id: str, idx: int, env_kwargs=dict(), record_video_path: str = None, wrappers=[], record_episode_kwargs=dict()):
    def _init():
        env = gym.make(env_id, disable_env_checker=True, **env_kwargs)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        for wrapper in wrappers:
            env = wrapper(env)
        
        if record_video_path is not None and (not record_episode_kwargs["record_single"] or idx == 0):
            env = RecordEpisodeWrapper(
                env,
                record_video_path,
                trajectory_name=f"trajectory_{idx}",
                video_fps=env.unwrapped.control_freq,
                **record_episode_kwargs
            )
        return env

    return _init

def env_factory_gpu(env_id: str, num_envs: int, seed: int, ignore_terminations=False, auto_reset=True, env_kwargs=dict(), record_video_path: str = None, wrappers=[], record_episode_kwargs=dict()):
    env = gym.make(env_id, num_envs=num_envs, disable_env_checker=True, **env_kwargs)
    for wrapper in wrappers:
        env = wrapper(env)
    if record_video_path is not None:
        env = RecordEpisodeWrapper(
            env,
            record_video_path,
            trajectory_name="trajectory",
            video_fps=env.unwrapped.control_freq,
            **record_episode_kwargs
        )
    env = ManiSkillVectorEnv(env, auto_reset=auto_reset, ignore_terminations=ignore_terminations, record_metrics=True)
    return env