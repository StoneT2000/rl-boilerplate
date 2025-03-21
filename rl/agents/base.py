class BasePolicy:
    def __init__(
        self,
        env_type: str,
        env=None,
        eval_env=None,
        num_envs: int = 1,
        num_eval_envs: int = 1,
        logger_cfg: LoggerConfig = None,
    ) -> None:
        """
        Base class for a policy

        Equips it with loopers and loggers
        """
        assert env is not None
        self.env_type = env_type
        self.jax_env = env_type == "jax"
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.setup_envs(env, eval_env)
        self.obs_shape = get_obs_shape(self.observation_space)
        self.action_dim = get_action_dim(self.action_space)

        # auto generate an experiment name based on the environment name and current time
        if logger_cfg is not None:
            if logger_cfg.exp_name is None:
                exp_name = f"{round(time.time_ns() / 1000)}"
                if hasattr(env, "name"):
                    exp_name = f"{env.name}/{exp_name}"
                logger_cfg.exp_name = exp_name
            if not logger_cfg.best_stats_cfg:
                logger_cfg.best_stats_cfg = {"test/ep_ret_avg": 1, "train/ep_ret_avg": 1}
            if logger_cfg.save_fn is None:
                logger_cfg.save_fn = self.save
            self.logger = Logger.create_from_cfg(logger_cfg)
        else:
            self.logger = None