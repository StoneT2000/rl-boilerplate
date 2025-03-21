import gymnasium as gym

def find_max_episode_steps_value(env):
    """Finds the max episode steps parameter given by user or registered in the environment.

    This is a useful utility as not all specs may include max episode steps and some wrappers
    may need access to this in order to implement e.g. TimeLimits correctly on the GPU sim."""
    cur = env
    if isinstance(cur, gym.vector.SyncVectorEnv):
        cur = env.envs[0]
    elif isinstance(cur, gym.vector.AsyncVectorEnv):
        raise NotImplementedError(
            "Currently cannot get max episode steps of an environment wrapped with gym.vector.AsyncVectorEnv"
        )
    while cur is not None:
        try:
            return cur.get_wrapper_attr("max_episode_steps")
        except AttributeError:
            pass
        try:
            return cur.get_wrapper_attr("_max_episode_steps")
        except AttributeError:
            pass
        if cur.spec is not None and cur.spec.max_episode_steps is not None:
            return cur.spec.max_episode_steps
        if hasattr(cur, "env"):
            cur = cur.env
        elif hasattr(cur, "_env"):
            cur = cur._env
        else:
            cur = None
    return None