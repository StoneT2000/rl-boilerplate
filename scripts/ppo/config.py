from rl.agents.ppo.config import PPOConfig
from rl.envs.make_env.make_env import EnvConfig
from rl.logger.logger import LoggerConfig
from rl.models.types import NetworkConfig
from dataclasses import dataclass, field

@dataclass
class PPONetworkConfig:
    shared_backbone: NetworkConfig | None = None
    actor: NetworkConfig = field(default_factory=NetworkConfig)
    critic: NetworkConfig = field(default_factory=NetworkConfig)

@dataclass
class PPOTrainConfig:
    seed: int
    """seed for the experiment"""
    env: EnvConfig
    """environment configurations"""
    eval_env: EnvConfig
    """evaluation environment configurations. You should only modify at most num_eval_envs, ignore_terminations, record_video_path, record_episode_kwargs. The rest are copied from env.env_kwargs"""
    network: PPONetworkConfig = field(default_factory=PPONetworkConfig)
    """actor critic neural net configurations"""
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    """logger configurations"""
    ppo: PPOConfig = field(default_factory=PPOConfig)
    """ppo hyperparameters"""

    total_timesteps: int = 100_000_000
    """total timesteps to train for"""
    eval_freq: int = 25
    """evaluation frequency in terms of training iterations"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    checkpoint: str | None = None
    """path to a pretrained checkpoint file to start training from"""



    cudagraphs: bool = False
    """whether to use cudagraphs"""
    compile: bool = False
    """whether to use torch.compile"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    def __post_init__(self):
        self.eval_env.env_id = self.env.env_id

        # self.eval_env.env_kwargs = dict(**self.env.env_kwargs)
        # any kwargs in env.env_kwargs that are not in eval_env.env_kwargs will be added to eval_env.env_kwargs
        for k, v in self.env.env_kwargs.items():
            if k not in self.eval_env.env_kwargs:
                self.eval_env.env_kwargs[k] = v
        
        self.eval_env.record_episode_kwargs["max_steps_per_video"] = self.num_eval_steps

try:
    import mani_skill.envs
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    ms3_configs = {
        "ms3-state": (
            "ManiSkill3 State based PPO training",
            PPOTrainConfig(
                seed=0,
                env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=4096,
                    vectorization_method="gpu",
                    ignore_terminations=False, # partial resets
                    env_kwargs=dict(
                        obs_mode="state",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=0,
                        # not sure how to permit other changes to this in CLI, waiting on https://github.com/brentyi/tyro/issues/277
                    )
                ),
                eval_env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=16,
                    vectorization_method="gpu",
                    ignore_terminations=True,
                    env_kwargs=dict(
                        obs_mode="state",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=1,
                        render_mode="rgb_array",
                        human_render_camera_configs=dict(shader_pack="default")
                    ),
                    record_video_path="videos",
                    record_episode_kwargs=dict(
                        save_trajectory=False,
                    )
                ),
                network=PPONetworkConfig(
                    actor=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="tanh",
                            output_activation="tanh",
                        ),
                    ),
                    critic=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[256, 256, 256],
                            activation="tanh",
                            output_activation="tanh",
                        ),
                    )
                ),
                num_steps=4,
                total_timesteps=10_000_000,
            )
        ),
        "ms3-rgb": (
            "ManiSkill3 RGB based PPO training",
            PPOTrainConfig(
                seed=0,
                env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=1024,
                    vectorization_method="gpu",
                    ignore_terminations=False, # partial resets
                    env_kwargs=dict(
                        obs_mode="rgb",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=0,
                        # not sure how to permit other changes to this in CLI, waiting on https://github.com/brentyi/tyro/issues/277
                    ),
                    wrappers=[FlattenRGBDObservationWrapper]
                ),
                eval_env=EnvConfig(
                    env_id="PickCube-v1",
                    num_envs=16,
                    vectorization_method="gpu",
                    ignore_terminations=True,
                    env_kwargs=dict(
                        obs_mode="rgb",
                        sim_backend="physx_cuda",
                        reconfiguration_freq=1,
                        render_mode="all",
                        human_render_camera_configs=dict(shader_pack="default")
                    ),
                    record_video_path="videos",
                    record_episode_kwargs=dict(
                        save_trajectory=False,
                    ),
                    wrappers=[FlattenRGBDObservationWrapper]
                ),
                network=PPONetworkConfig(
                    shared_backbone=NetworkConfig(
                        type="nature_cnn",
                        arch_cfg=dict(
                            state_features=[256],
                            activation="relu",
                            output_activation="relu",
                        ),
                    ),
                    actor=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[512],
                            activation="relu",
                            output_activation="relu",
                        ),
                    ),
                    critic=NetworkConfig(
                        type="mlp",
                        arch_cfg=dict(
                            features=[512],
                            activation="relu",
                            output_activation="relu",
                        ),
                    )
                ),
                num_steps=16,
                total_timesteps=20_000_000,
            )
        )
    }
except:
    pass